/*
 * Sequential implementation of the Conjugate Gradient Method.
 *
 * Authors : Lilia Ziane Khodja & Charles Bouillaguet
 *
 * v1.02 (2020-04-3)
 *
 * CHANGE LOG:
 *    v1.01 : fix a minor printing bug in load_mm (incorrect CSR matrix size)
 *    v1.02 : use https instead of http in "PRO-TIP"
 *
 * USAGE:
 * 	$ ./cg --matrix bcsstk13.mtx                # loading matrix from file
 *      $ ./cg --matrix bcsstk13.mtx > /dev/null    # ignoring solution
 *	$ ./cg < bcsstk13.mtx > /dev/null           # loading matrix from stdin
 *      $  zcat matrix.mtx.gz | ./cg                # loading gziped matrix from
 *      $ ./cg --matrix bcsstk13.mtx --seed 42      # changing right-hand side
 *      $ ./cg --no-check < bcsstk13.mtx            # no safety check
 *
 * PRO-TIP :
 *      # downloading and uncompressing the matrix on the fly
 *	$ curl --silent https://hpc.fil.cool/matrix/bcsstk13.mtx.gz | zcat | ./cg
 */

 // code
 // !# Ce qu'on a modifié/ajouté
 // !#!  Àmodifier

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <math.h>
#include <getopt.h>
#include <sys/time.h>
#include <mpi.h>
#include "mmio.h"

#define MASTER 0
#define THRESHOLD 1e-8		// maximum tolerance threshold
double maxerrr;
struct csr_matrix_t {
	int n;			// dimension
	int nz;			// number of non-zero entries
	int *Ap;		// row pointers
	int *Aj;		// column indices
	double *Ax;		// actual coefficient
};

/*************************** Utility functions ********************************/

/* Seconds (wall-clock time) since an arbitrary point in the past */
double wtime()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (double)ts.tv_sec + ts.tv_usec / 1e6;
}

/* Pseudo-random function to initialize b (rumors says it comes from the NSA) */
#define ROR(x, r) ((x >> r) | (x << (64 - r)))
#define ROL(x, r) ((x << r) | (x >> (64 - r)))
#define R(x, y, k) (x = ROR(x, 8), x += y, x ^= k, y = ROL(y, 3), y ^= x)
double PRF(int i, unsigned long long seed)
{
	unsigned long long y = i, x = 0xBaadCafe, b = 0xDeadBeef, a = seed;
	R(x, y, b);
	for (int i = 0; i < 31; i++) {
		R(a, b, i);
		R(x, y, b);
	}
	x += i;
	union { double d; unsigned long long l;	} res;
	res.l = ((x << 2) >> 2) | (((1 << 10) - 1ll) << 52);
	return 2 * (res.d - 1.5);
}

/*************************** Matrix IO ****************************************/

/* Load MatrixMarket sparse symetric matrix from the file descriptor f */
struct csr_matrix_t *load_mm(FILE * f, int my_rank)
{
	MM_typecode matcode;
	int n, m, nnz;

	/* -------- STEP 1 : load the matrix in COOrdinate format */
	double start = wtime();

	/* read the header, check format */
	if (mm_read_banner(f, &matcode) != 0)
		errx(1, "Could not process Matrix Market banner.\n");
	if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode))
		errx(1, "Matrix Market type: [%s] not supported (only sparse matrices are OK)", mm_typecode_to_str(matcode));
	if (!mm_is_symmetric(matcode) || !mm_is_real(matcode))
		errx(1, "Matrix type [%s] not supported (only real symmetric are OK)", mm_typecode_to_str(matcode));
	if (mm_read_mtx_crd_size(f, &n, &m, &nnz) != 0)
		errx(1, "Cannot read matrix size");
	if(my_rank==MASTER) // !#
		fprintf(stderr, "[IO] Loading [%s] %d x %d with %d nz in triplet format\n", mm_typecode_to_str(matcode), n, n, nnz);
	if(my_rank==MASTER) // !#
		fprintf(stderr, "     ---> for this, I will allocate %.1f MByte\n", 1e-6 * (40.0 * nnz + 8.0 * n));

	/* Allocate memory for the COOrdinate representation of the matrix (lower-triangle only) */
	int *Ti = malloc(nnz * sizeof(*Ti));
	int *Tj = malloc(nnz * sizeof(*Tj));
	double *Tx = malloc(nnz * sizeof(*Tx));
	if (Ti == NULL || Tj == NULL || Tx == NULL)
		err(1, "Cannot allocate (triplet) sparse matrix");

	/* Parse and load actual entries */
	for (int u = 0; u < nnz; u++) {
		int i, j;
		double x;
		if (3 != fscanf(f, "%d %d %lg\n", &i, &j, &x))
			errx(1, "parse error entry %d\n", u);
		Ti[u] = i - 1;	/* MatrixMarket is 1-based */
		Tj[u] = j - 1;
		/*
		 * Uncomment this to check input (but it slows reading)
		 * if (i < 1 || i > n || j < 1 || j > i)
		 *	errx(2, "invalid entry %d : %d %d\n", u, i, j);
		 */
		Tx[u] = x;
	}

	double stop = wtime();
	if(my_rank==MASTER) // !#
		fprintf(stderr, "     ---> loaded in %.1fs\n", stop - start);

	/* -------- STEP 2: Convert to CSR (compressed sparse row) representation ----- */
	start = wtime();

	/* allocate CSR matrix */
	struct csr_matrix_t *A = malloc(sizeof(*A));
	if (A == NULL)
		err(1, "malloc failed");
	int *w = malloc((n + 1) * sizeof(*w));
	int *Ap = malloc((n + 1) * sizeof(*Ap));
	int *Aj = malloc(2 * nnz * sizeof(*Ap));
	double *Ax = malloc(2 * nnz * sizeof(*Ax));
	if (w == NULL || Ap == NULL || Aj == NULL || Ax == NULL)
		err(1, "Cannot allocate (CSR) sparse matrix");

	/* the following is essentially a bucket sort */

	/* Count the number of entries in each row */
	for (int i = 0; i < n; i++)
		w[i] = 0;
	for (int u = 0; u < nnz; u++) {
		int i = Ti[u];
		int j = Tj[u];
		w[i]++;
		if (i != j)	/* the file contains only the lower triangular part */
			w[j]++;
	}

	/* Compute row pointers (prefix-sum) */
	int sum = 0;
	for (int i = 0; i < n; i++) {
		Ap[i] = sum;
		sum += w[i];
		w[i] = Ap[i];
	}
	Ap[n] = sum;

	/* Dispatch entries in the right rows */
	for (int u = 0; u < nnz; u++) {
		int i = Ti[u];
		int j = Tj[u];
		double x = Tx[u];
		Aj[w[i]] = j;
		Ax[w[i]] = x;
		w[i]++;
		if (i != j) {	/* off-diagonal entries are duplicated */
			Aj[w[j]] = i;
			Ax[w[j]] = x;
			w[j]++;
		}
	}

	/* release COOrdinate representation */
	free(w);
	free(Ti);
	free(Tj);
	free(Tx);
	stop = wtime();
	if(my_rank==MASTER) // !#
		fprintf(stderr, "     ---> converted to CSR format in %.1fs\n", stop - start);
	if(my_rank==MASTER) // !#
		fprintf(stderr, "     ---> CSR matrix size = %.1fMbyte\n", 1e-6 * (24. * nnz + 4. * n));

	A->n = n;
	A->nz = sum;
	A->Ap = Ap;
	A->Aj = Aj;
	A->Ax = Ax;
	return A;
}

/*************************** Matrix accessors *********************************/

/* Copy the diagonal of A into the vector d. */
void extract_diagonal(const struct csr_matrix_t *A, double *d)
{
	int n = A->n;
	int *Ap = A->Ap;
	int *Aj = A->Aj;
	double *Ax = A->Ax;
	for (int i = 0; i < n; i++) {
		d[i] = 0.0;
		for (int u = Ap[i]; u < Ap[i + 1]; u++)
			if (i == Aj[u])
				d[i] += Ax[u];
	}
}

/* Matrix-vector product (with A in CSR format) : y = Ax */
void sp_gemv(const struct csr_matrix_t *A, const double *x, double *y)
{
	int n = A->n;
	int *Ap = A->Ap;
	int *Aj = A->Aj;
	double *Ax = A->Ax;
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		y[i] = 0;
		for (int u = Ap[i]; u < Ap[i + 1]; u++) {
			int j = Aj[u];
			double A_ij = Ax[u];
			y[i] += A_ij * x[j];
		}
	}
}

// !# Une version locale de sp_gemv, utile pour mpi
void sp_gemv_local(const struct csr_matrix_t *A, const double *x, double *y_local, int n_local, int pos_local)
{
    int *Ap = A->Ap;
    int *Aj = A->Aj;
    double *Ax = A->Ax;

	// !# On va de pos_local à pos_local + n_local plutôt que de traiter tous les indices
    #pragma omp parallel for
    for (int i = pos_local; i < pos_local + n_local; i++) {
            y_local[i-pos_local] = 0;
        for (int u = Ap[i]; u < Ap[i + 1]; u++) {
                int j = Aj[u];
                double A_ij = Ax[u];
                y_local[i-pos_local] += A_ij * x[j];
        }
    }
}

/*************************** Vector operations ********************************/

/* dot product */
double dot(const int n, const double *x, const double *y)
{
	double sum = 0.0;
	#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < n; i++)
		sum += x[i] * y[i];
	return sum;
}

/* euclidean norm (a.k.a 2-norm) */
double norm(const int n, const double *x)
{
	return sqrt(dot(n, x, x));
}

/*********************** conjugate gradient algorithm *************************/
/* Solve Ax == b (the solution is written in x). Scratch must be preallocated of size 6n */
void cg_solve(const struct csr_matrix_t *A, const double *b, double *x, const double epsilon, double *scratch, int my_rank, int np)
{
	int n = A->n;
	int nz = A->nz;

	// !# Seul le MASTER = 0 affiche ces infos
	if(my_rank==MASTER)
	{
		fprintf(stderr, "[CG] Starting iterative solver\n");
		fprintf(stderr, "     ---> Using : %d nodes\n",np);
		fprintf(stderr, "     ---> Working set : %.1fMbyte\n", 1e-6 * (12.0 * nz + 52.0 * n));
		fprintf(stderr, "     ---> Per iteration: %.2g FLOP in sp_gemv() and %.2g FLOP in the rest\n", 2. * nz, 12. * n);
	}

	// !# On calcule les indices pour tout le monde
	int *recvcounts = malloc(np*sizeof(int)); // Tailles des vecteurs de chaque noeud
	int *displs = malloc(np*sizeof(int)); // Positions des vecteurs de chaque noeud
	for(int i = 0 ; i < np ; i++){
		recvcounts[i] = (i+1)*n/np - i*n/np;
		displs[i] = i*n/np;
	}

	// !# On en déduit les bons indices du processeur courant
	int n_local = recvcounts[my_rank];
	int start_pos = displs[my_rank];
	int end_pos = displs[my_rank] + recvcounts[my_rank];


	// !# On demande la mémoire pour les vecteurs locaux
	double *x_local = malloc(n_local * sizeof(double)); // Résultat
	double *r_local = malloc(n_local * sizeof(double)); // residue
	double *z_local = malloc(n_local * sizeof(double)); // preconditioned-residue
	double *p_local = malloc(n_local * sizeof(double)); // search direction
	double *q_local = malloc(n_local * sizeof(double)); // q == Ap

	// !# On les affiche pour vérifier
	fprintf(stderr,"!###  Noeud(%d) : de %d à %d : taille %d \n",my_rank,start_pos,end_pos,n_local);

	double *p = scratch;	// search direction
	double *d = scratch +  n;	// diagonal entries of A (Jacobi preconditioning)

	/* Isolate diagonal */
	// !# Tous les processeurs font les calculs initiaux extract diagonal
	extract_diagonal(A, d);

	/*
	 * This function follows closely the pseudo-code given in the (english)
	 * Wikipedia page "Conjugate gradient method". This is the version with
	 * preconditionning.
	 */

	/* We use x == 0 --- this avoids the first matrix-vector product. */
	// Ce calcul initial n'est fait qu'une seule fois et n'a donc pas besoin d'être parrallélisé
	for (int i = 0; i < n; i++)	// p <-- z
		p[i] = b[i]/d[i];

	// !# On initialise ensuite les vecteurs locaux qui en ont besoin
	for (int i = start_pos; i < end_pos; i++)
		r_local[i-start_pos] = b[i];
	for (int i = start_pos; i < end_pos; i++)
		p_local[i-start_pos] = p[i];
	for (int i = start_pos; i < end_pos; i++)
		z_local[i-start_pos] = r_local[i-start_pos]/d[i];
	for (int i = start_pos; i < end_pos; i++)
		p_local[i-start_pos] = p[i];

	// !# On parrallélise la somme pour éviter d'avoir à récupérer r et z en entier
	double rz;
	double rz_local = dot(n_local,r_local,z_local);
	MPI_Allreduce( &rz_local,&rz,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);


	// !# Idem pour r
	double err_2_local = norm(n_local, r_local);
	double err_2 = 0.0;
	MPI_Allreduce( &err_2_local,&err_2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	double err_actuelle = sqrt(err_2); // !# Erreur actuelle

	double start = wtime();
	double last_display = start;
	int iter = 0;


	while (err_actuelle> epsilon) {
		/* loop invariant : rz = dot(r, z) */
		double old_rz = rz;

		// !# sp_gemv est distribué : l'entrée p est mise en commun à la fin de la boucle
		sp_gemv_local(A, p, q_local,n_local,start_pos);	/* q <-- A.p */

		// !# On distribue le produit
		double pq_local = dot(n_local, p_local,q_local);
		double pq = 0.0;
		MPI_Allreduce( &pq_local,&pq,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); //faire réduction de cette somme pour tous les processeurs

		double alpha = old_rz / pq;

		// !# On distribue les calculs élémentaires :
		
		#pragma omp parallel for
		for (int i = start_pos; i < end_pos; i++)	// x <-- x + alpha*p
		    x_local[i-start_pos] = x_local[i-start_pos] + alpha * p_local[i-start_pos];
		// !# inutile de gather x maintenant car les autres ne l'utilisent pas
		
		#pragma omp parallel for
		for (int i = start_pos; i < end_pos; i++)	// r <-- r - alpha*q
		    r_local[i-start_pos] = r_local[i-start_pos] - alpha * q_local[i-start_pos];
		// !# inutile de gather r maintenant car les autres ne l'utilisent pas
		
		#pragma omp parallel for
		for (int i = start_pos; i < end_pos; i++)	// z <-- M^(-1).r
		    z_local[i-start_pos] = r_local[i-start_pos] / d[i];
		// !# inutile de gather z maintenant car les autres ne l'utilisent pas

		// !# On ditribue le produit pour rz:
		rz_local = dot(n_local,r_local,z_local);	// restore invariant
		MPI_Allreduce( &rz_local,&rz,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

		double beta = rz / old_rz;

		// !# On distribue ces calculs élémentaires :
		#pragma omp parallel for
		for (int i = start_pos; i < end_pos; i++)	// p <-- z + beta*p
			p_local[i-start_pos] = z_local[i-start_pos] + beta * p_local[i-start_pos];

		// !# Le p est le seul qu'on doit rassembler car on l'utilise en entier dans sp_gemv
		MPI_Allgatherv(p_local,n_local , MPI_DOUBLE,p,recvcounts,displs , MPI_DOUBLE, MPI_COMM_WORLD);

		iter++;
		double t = wtime();

		// !# On distribue le calcul de l'erreur
		err_2_local = norm(n_local, r_local);
		err_2 = 0.0;
		MPI_Allreduce( &err_2_local,&err_2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); //faire réduction de cette somme pour tous les processeurs
		err_actuelle = sqrt(err_2);

		// !# Seul le 0 affiche ces infos
		if (t - last_display > 0.5 && my_rank==MASTER) {
			/* verbosity */
			double rate = iter / (t - start);	// iterations per s.
			double GFLOPs = 1e-9 * rate * (2 * nz + 12 * n);
			// fprintf(stderr, "\r     ---> error : %2.2e, iter : %d (%.1f it/s, %.2f GFLOPs)", err_actuelle , iter, rate, GFLOPs);
			fprintf(stderr, "\r     ---> error : %2.2e, iter : %d (%.1f it/s, %.2f GFLOPs)", err_actuelle, iter, rate, GFLOPs);
			fflush(stdout);
			last_display = t;
		}
	}
	// !# On rassemble x maintenant que tout est calculé
	MPI_Allgatherv(x_local,n_local , MPI_DOUBLE,x,recvcounts,displs , MPI_DOUBLE, MPI_COMM_WORLD);
	// !# Seul le 0 affiche cette info
	if(my_rank==MASTER)
		fprintf(stderr, "\n     ---> Finished in %.1fs and %d iterations\n", wtime() - start, iter);

}

/******************************* main program *********************************/

/* options descriptor */
struct option longopts[6] = {
	{"seed", required_argument, NULL, 's'},
	{"rhs", required_argument, NULL, 'r'},
	{"matrix", required_argument, NULL, 'm'},
	{"solution", required_argument, NULL, 'o'},
	{"no-check", no_argument, NULL, 'c'},
	{NULL, 0, NULL, 0}
};

int main(int argc, char **argv)
{
	fprintf(stderr, "ENTREE\n");
	maxerrr = 0;
	// !# Réception des indices
	int my_rank, np;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&np);

	fprintf(stderr, "Je suis le %d sur %d\n", my_rank,np);

	/* Parse command-line options */
	long long seed = 0;
	char *rhs_filename = NULL;
	char *matrix_filename = NULL;
	char *solution_filename = NULL;
	int safety_check = 1;
	char ch;

	while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
		switch (ch) {
		case 's':
			seed = atoll(optarg);
			break;
		case 'r':
			rhs_filename = optarg;
			break;
		case 'm':
			matrix_filename = optarg;
			break;
		case 'o':
			solution_filename = optarg;
			break;
		case 'c':
			safety_check = 0;
			break;
		default:
			errx(1, "Unknown option");
		}
	}

	/* Load the matrix */
	FILE *f_mat = stdin;
	if (matrix_filename) {
		f_mat = fopen(matrix_filename, "r");
		if (f_mat == NULL)
			err(1, "cannot matrix file %s", matrix_filename);
	}
	struct csr_matrix_t *A = load_mm(f_mat,my_rank);

	/* Allocate memory */
	int n = A->n;
	// !# Plus besoin de q, r et z donc 7-3 =4
	double *mem = malloc(4 * n * sizeof(double));
	if (mem == NULL)
		err(1, "cannot allocate dense vectors");
	double *x = mem;	/* solution vector */
	double *b = mem + n;	/* right-hand side */
	double *scratch = mem + 2 * n;	/* workspace for cg_solve() */

	/* Prepare right-hand size */
	if (rhs_filename) {	/* load from file */
		FILE *f_b = fopen(rhs_filename, "r");
		if (f_b == NULL)
			err(1, "cannot open %s", rhs_filename);
		if(my_rank==MASTER) // !#
			fprintf(stderr, "[IO] Loading b from %s\n", rhs_filename);
		for (int i = 0; i < n; i++) {
			if (1 != fscanf(f_b, "%lg\n", &b[i]))
				errx(1, "parse error entry %d\n", i);
		}
		fclose(f_b);
	} else {
		for (int i = 0; i < n; i++)
			b[i] = PRF(i, seed);
	}

	/* solve Ax == b */
	// !# Ajout en arguments de son rang et du nombre de processeurs
	cg_solve(A, b, x, THRESHOLD, scratch,my_rank, np);

	// !# Seul le processeur 0 devra exécuter la suite :
	if(my_rank == MASTER)
	{
		/* Check result */
		if (safety_check) {
			double *y = scratch;
			sp_gemv(A, x, y);	// y = Ax
			for (int i = 0; i < n; i++)	// y = Ax - b
				y[i] -= b[i];
			fprintf(stderr, "[check] max error = %2.2e\n", norm(n, y));
		}

		/* Dump the solution vector */
		FILE *f_x = stdout;
		if (solution_filename != NULL) {
			f_x = fopen(solution_filename, "w");
			if (f_x == NULL)
				err(1, "cannot open solution file %s", solution_filename);
			fprintf(stderr, "[IO] writing solution to %s\n", solution_filename);
		}
		for (int i = 0; i < n; i++)
			fprintf(f_x, "%a\n", x[i]);
	}
	MPI_Finalize();
	return EXIT_SUCCESS;
}
