
/* 	$ ./cg --matrix bcsstk13.mtx                # loading matrix from file
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
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <math.h>
#include <getopt.h>
#include <sys/time.h>
#include <mpi.h>
#include "mmio.h"
#define SIZE_H_N 50
#define THRESHOLD 1e-8		// maximum tolerance threshold
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
struct csr_matrix_t *load_mm(FILE * f,int my_rank)
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
	if(my_rank==0){
	fprintf(stderr, "[IO] Loading [%s] %d x %d with %d nz in triplet format\n", mm_typecode_to_str(matcode), n, n, nnz);
	fprintf(stderr, "     ---> for this, I will allocate %.1f MByte\n", 1e-6 * (40.0 * nnz + 8.0 * n));
	}
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
	if(my_rank==0){
	fprintf(stderr, "     ---> loaded in %.1fs\n", stop - start);
}
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
	if (my_rank==0){
	fprintf(stderr, "     ---> converted to CSR format in %.1fs\n", stop - start);
	fprintf(stderr, "     ---> CSR matrix size = %.1fMbyte\n", 1e-6 * (24. * nnz + 4. * n));
	}
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
	for (int i = 0; i < n; i++) {
		y[i] = 0;
		for (int u = Ap[i]; u < Ap[i + 1]; u++) {
			int j = Aj[u];
			double A_ij = Ax[u];
			y[i] += A_ij * x[j];
		}
	}
}

void sp_gemv_mpi(const struct csr_matrix_t *A, const double *x, double *y_local,int taille_loc, int debut)
{
	int *Ap = A->Ap;
	int *Aj = A->Aj;
	double *Ax = A->Ax;

	for (int i = debut; i < debut+taille_loc; i++) {
		y_local[i-debut] = 0;
		for (int u = Ap[i]; u < Ap[i + 1]; u++) {
			int j = Aj[u];
			double A_ij = Ax[u];
			y_local[i-debut] += A_ij * x[j];
		}
	}

}



/*************************** Vector operations ********************************/

/* dot product */
double dot_local(const int n, const double *x, const double *y)
{
	double sum = 0.0;
	for (int i = 0; i < n; i++)
		sum += x[i] * y[i];
	return sum;
}


/* euclidean norm (a.k.a 2-norm) */
double norm(const int n, const double *x)
{
	return sqrt(dot_local(n, x, x));
}


/***********************  *************************/

/* Solve Ax == b (the solution is written in x). Scratch must be preallocated of size 6n */
void cg_solve_mpi(const struct csr_matrix_t *A, const double *b, double *x, const double epsilon, double *scratch,int my_rank, int total)
{
	int n = A->n; /// length of matrix
	int nz = A->nz; //number of non zeros


	if(my_rank==0){
	fprintf(stderr, "[CG] Starting iterative solver\n");
	fprintf(stderr, "     ---> Working set : %.1fMbyte\n", 1e-6 * (12.0 * nz + 52.0 * n));
	fprintf(stderr, "     ---> Per iteration: %.2g FLOP in sp_gemv() and %.2g FLOP in the rest\n", 2. * nz, 12. * n);
	}




	// Vector size and displacement for each processor
	int *taille_local=malloc(total*sizeof(int));
	int *deplac_local=malloc(total*sizeof(int));
	for(int i=0; i<total;i++){
		taille_local[i] = (i+1)*n/total- i*n/total;
		deplac_local[i] = i*n/total;
	}




	int taille_loc=	taille_local[my_rank];
	int debut=deplac_local[my_rank];
	int fin=deplac_local[my_rank]+taille_local[my_rank];
	/////Les matrice local à utiliser
	double *r_local = malloc(taille_loc*sizeof(double));	        // residue
	double *z_local = malloc(taille_loc*sizeof(double));	// preconditioned-residue
	double *p_local = malloc(taille_loc*sizeof(double));	// search direction
	double *q_local = malloc(taille_loc*sizeof(double));	// q == Ap
	double *x_local = malloc(taille_loc*sizeof(double));
	fprintf(stderr,"!###  Noeud(%d) : de %d à %d : taille %d \n",my_rank,debut,fin,taille_loc);

	//////////// ON GARDE P ET D
	double *p = scratch ;	// search direction
	double *d = scratch +n;	// diagonal entries of A (Jacobi preconditioning)

	/* Isolate diagonal */
	extract_diagonal(A, d);

	/* We use x == 0 --- this avoids the first matrix-vector product.*/
	//On supprime x car pas besoin
	for (int i = 0; i < n; i++)
		p[i] = b[i]/d[i];

	for (int i =debut ; i < fin; i++)	// r <-- b - Ax == b
		r_local[i-debut] = b[i];
	for (int i = debut; i < fin; i++)	// p <-- z
		p_local[i-debut] = p[i];
	for (int i = debut; i < fin; i++)	// z <-- M^(-1).r
		z_local[i-debut] = r_local[i-debut] / d[i];
	for (int i = debut; i < fin; i++)	// p <-- z
		p_local[i-debut] = p[i];

	double rz;
	double rz_local = dot_local(taille_loc, r_local, z_local);
	MPI_Allreduce(&rz_local,&rz,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	double erreur_local=norm(taille_loc,r_local);
	double erreur2=0.0;
	MPI_Allreduce(&erreur_local,&erreur2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	double erreur=sqrt(erreur2);

	double start = wtime();
	double last_display = start;
	int iter = 0;


	while (erreur > epsilon) {
		/* loop invariant : rz = dot(r, z) */
		double old_rz = rz;

		sp_gemv_mpi(A, p, q_local,taille_loc,debut);	/* q <-- A.p */

		double dot=0.0;
		double local = dot_local(taille_loc, p_local, q_local);
		MPI_Allreduce(&local,&dot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

		double alpha = old_rz / dot;

		for (int i = debut; i < fin; i++){	// x <-- x + alpha*p
			x_local[i-debut] =x_local[i-debut]+ alpha * p_local[i-debut];}

		for (int i = debut; i < fin; i++){	// r <-- r - alpha*q
			r_local[i-debut]= r_local[i-debut]- alpha * q_local[i-debut];}

		for (int i = debut; i < fin; i++){	// z <-- M^(-1).r
			z_local[i-debut] =  r_local[i-debut] / d[i];
		}
		rz_local=dot_local(n, r_local, z_local);
		MPI_Allreduce(&rz_local,&rz,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		double beta = rz / old_rz;

		for (int i = debut; i < fin; i++)	// p <-- z + beta*p
			p_local[i-debut] = z_local[i-debut] + beta * p_local[i-debut];

		///On rassemble p car on en a besoin pour le produit matrice
		MPI_Allgatherv(p_local,taille_loc, MPI_DOUBLE,p,taille_local,deplac_local,MPI_DOUBLE, MPI_COMM_WORLD);
		iter++;
		double t = wtime();

		erreur_local=norm(taille_loc, r_local);
		erreur2=0.0;
		MPI_Allreduce(&erreur_local,&erreur2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		erreur=sqrt(erreur2);

		if (t - last_display > 0.5 && my_rank==0) {
			/* verbosity */
			double rate = iter / (t - start);	// iterations per s.
			double GFLOPs = 1e-9 * rate * (2 * nz + 12 * n);
			fprintf(stderr, "\r     ---> error : %2.2e, iter : %d (%.1f it/s, %.2f GFLOPs)", erreur, iter, rate, GFLOPs);
			fflush(stdout);
			last_display = t;
		}
	}

	MPI_Allgatherv(x_local,taille_loc,MPI_DOUBLE,x,taille_local,deplac_local,MPI_DOUBLE, MPI_COMM_WORLD);
	if (my_rank==0) {
		fprintf(stderr, "\n     ---> Finished in %.1fs and %d iterations\n", wtime() - start, iter);
	}

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
	/* Initializing MPI */
	int my_rank, total;

    //initialisation

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&total);


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
		if(my_rank==0){
			fprintf(stderr, "[IO] Loading b from %s\n", rhs_filename);}
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
	cg_solve_mpi(A, b, x, THRESHOLD, scratch,my_rank,total);

	/* Check result */
	if(my_rank==0){
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
