#!/usr/bin/env python3
import array
import subprocess
import json
import sys
import os
from hashlib import sha256
import base64

#################################################################

# Paramètres du système (ne pas modifier)
SERVER = "https://hpc.fil.cool"
params = {}
params['version'] = 3
params['proof'] = 0x65dec1542f679f51

# Paramètres du calcul (à adapter)
params['matrix'] = "bcsstk13"
params['users'] = ["Charles Bouillaguet", "Lilia Ziane Khodja"]

# Description du code exécuté
params['software'] = """Code séquentiel fourni.
Pas d'améliorations apportées."""

# Description du matériel utilisé pour l'exécution
params['nodes'] = 4   # nombre de noeuds
params['cores'] = 8   # nombre total de coeurs
params['hardware'] = """4 PCs de la Salle 327 (Esclangon).
Machines de bureau DELL équipés de CPU Intel i3 à 2 coeurs, 3Ghz, et 4Go de RAM."""

# Comment exécuter le solveur :
#   {matrix} sera remplacé par la valeur ci-dessus.
#   {nodes}  sera remplacé par la valeur ci-dessus.
#   {cores}  sera remplacé par la valeur ci-dessus.
#   {seed}   sera remplacé par la valeur fournie par le serveur.
#   On peut ajouter toutes les options qu'on veut, utiliser mpiexec, etc.
command_line = "./cg --matrix {matrix}.mtx --seed {seed}"
#command_line = "zcat matrices/{matrix}.mtx.gz | ./cg --seed {seed}"
#command_line = "mpiexec --n {cores} --hostfile nodes.txt --display-map ./cg --matrix {matrix}.mtx --seed {seed}"
#command_line = "mpiexec --n {nodes} -hostfile nodes.txt --map-by ppr:1:node ./cg --matrix {matrix}.mtx --seed {seed}"

######################### Main Program ###########################

def main():
    # vérifie que le script a bien été modifié !
    assert "Charles Bouillaguet" not in params['users']

    checkpoint_filename = "checkpoint-{}.json".format(params['matrix'])

    if not os.path.exists(checkpoint_filename):
        # pas de checkpoint : on fait le calcul.
        seed, x, e, secret, chunks = do_computation()
        # OK, maintenant on peut se relaxer et prendre notre temps. Le serveur sait
        # qu'on a fini. On a intérêt à sauvegarder notre état interne, juste au cas 
        # où un problème ait lieu.
        checkpoint(checkpoint_filename, seed, x, e)
    else:
        # reprise en cas de panne à partir du checkpoint
        seed, x, e, secret, chunks = restart(checkpoint_filename)

    # prouve au serveur qu'on a fait le calcul correctement, obtient un reçu, supprime le checkpoint
    do_proof(seed, x, e, secret, chunks)
    os.unlink(checkpoint_filename)


def do_computation():
    """
    Obtient un challenge auprès du serveur et lance le calcul.
    """
    print("--> [Connecting]")
    print()

    # On demande au serveur une graine aléatoire pour générer le membre droit
    d = request(SERVER + "/start", **params)
    print("SERVER says : {}".format(d['msg']))
    print("SERVER gives seed = {}".format(d['seed']))
    print()
    
    # Tente d'éxecuter le solveur. 
    cmd = command_line.format(seed=d['seed'], **params)
    print("--> [Starting : {}]".format(cmd))
    try:
        print()
        print("---------------------------------------------------------------")
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE)
        print("---------------------------------------------------------------")
        print()
    except subprocess.CalledProcessError as e:
        print("---------------------------------------------------------------")
        print("Ooops, le solveur a échoué (code de retour != 0).")
        sys.exit(1)
    
    # REMARQUE : la mesure de temps effectuée par le serveur inclut a priori le 
    #            temps nécessaire pour charger la matrice depuis le système de 
    #            fichier, ce qui est séquentiel et peut être assez long pour les
    #            plus grosses. 
    #            Le comité d'organisation du benchmark (cf. infra) a étudié la 
    #            possibilité que des étudiants puissent tenter de lancer le 
    #            solveur et de précharger la matrice en RAM **AVANT** d'avoir 
    #            reçu le vecteur b, pour avoir un avantage compétitif. Le comité
    #            a décidé que les étudiants n'étaient certainement pas assez 
    #            bons pour mettre en oeuvre cette stratégie, et d'ailleurs on ne
    #            va pas leur suggérer.
    
    # le solveur a terminé avec succès. On récupère le vecteur solution.
    x = []
    for line in result.stdout.decode().splitlines():
        x.append(float.fromhex(line))
    print("Got solution from the solver ({:.1f} Kbyte).".format(0.0078125 * len(x)))
    print()

    # Le chrono tourne toujours. On hashe x le plus vite possible, puis on 
    # envoie l'empreinte au serveur pour mettre le vecteur en gage.
    print("--> [Computing commitment]")
    print()
    xbytes = array.array('d', x).tobytes()
    chunks = [xbytes[i:i + 304] for i in range(0, len(xbytes), 304)]
    commitment, secret = commit(chunks)

    print("--> [Sending]")
    print()

    e = request(SERVER + "/stop", stuff=d['stuff'], commitment=commitment, xsize=len(x))
    print("SERVER says : {}".format(e['msg']))
    print('SERVER wants me to transmit {} "random" coefficients of x (out of {})'.format(len(e['transmit']), len(x)))
    print("SERVER wants me to decommit x[{}]".format(e['challenge']))
    print()
    
    return d['seed'], x, e, secret, chunks



def checkpoint(filename, seed, x, e):
    """
    Sauvegarde l'état actuel du processus.
    """
    print("--> [Checkpointing]")
    with open(filename, "w") as f:
        json.dump([seed, x, e], f)
    print()


def restart(filename):
    """
    Reprise à partir du checkpoint
    """
    print("--> [Warm start from checkpoint]")
    with open(filename, "r") as f:
        seed, x, e = json.load(f)
    print("--> [Computing commitment]")
    xbytes = array.array('d', x).tobytes()
    chunks = [xbytes[i:i + 304] for i in range(0, len(xbytes), 304)]
    _, secret = commit(chunks)
    return seed, x, e, secret, chunks


def do_proof(seed, x, e, secret, chunks):
    # Maintenant, pour obtenir un reçu, il faut convaincre le serveur qu'on a 
    # effectué le calcul correctement. On pourrait envoyer tout x, mais c'est 
    # trop long (il peut faire des dizaines de Mo).
    # Pour régler ce problème, une réunion au sommet s'est tenue entre les 
    # responsables de HPC et de ISEC (le fameux "comité d'organisation du 
    # benchmark"). Il est en sorti un protocole (non-publié) pour démontrer au 
    # serveur qu'on a VRAIMENT calculé une solution de Ax == b en ne lui 
    # révélant qu'un tout petit bout de x. Comme le serveur ne vérifie pas 
    # tout x, il doit bien y avoir un moyen de tricher... peut-être ?
    coeffs = [x[i] for i in e['transmit']]
    proof = decommit(secret, chunks, e['challenge'] // 38)
    size = len(json.dumps([coeffs, proof]))

    print("--> [Sending proof to the server (approx {} bytes)]".format(size))
    f = request(SERVER + "/proof", stuff=e['stuff'], coeffs=coeffs, response=proof)
    print()
    print("SERVER says : {}".format(f['msg']))
    print()

    # YES ! On a obtenu le reçu du serveur. On le sauvegarde précieusement.
    receipt_filename = "{}-{}.receipt".format(params['matrix'], seed)
    print("--> [writing receipt in {}]".format(receipt_filename))
    with open(receipt_filename, "w") as file:
        json.dump(f['receipt'], file)
    print()


##################################################################
# Fonctions auxiliaires

def request(url, method='GET', **kwds):
    """
    Soumet une requête HTTP au serveur de benchmark.
    """
    import urllib.request
    data = json.dumps(kwds).encode()
    try:
        req = urllib.request.Request(url, method)
        with urllib.request.urlopen(req, data) as connexion:
            response = connexion.read()
    except urllib.error.HTTPError as e:
        print("SERVER ERROR : {}".format(e.read().decode()))
        print("Aborting.")
        sys.exit(1)
    return json.loads(response)

def commit(it):
    """ 
    Prend en entrée un iterable d'objets de type bytes(), et renvoie une
    paire (mise en gage, secret). Le secret est ensuite utilisé par decommit().
    """
    stack = {}
    def H(x):
        return sha256(x).digest()
    def push(item, level=0):
        item_hash, _ = item
        while level in stack:
            pair = stack[level]
            pair_hash, _ = pair
            del stack[level]
            item_hash = H(b'10' + pair_hash + item_hash)
            item = item_hash, [pair, item]
            level += 1
        stack[level] = item
    for x in it:
        push((H(b'00' + x), []))
    while len(stack) > 1:
        push((H(b'01'), []), min(stack))
    _, top = stack.popitem()
    top_hash, _ = top
    return H(b'11' + top_hash).hex(), [top]


def decommit(secret, x, i):
    """
    Étant donné le secret créé lors de la mise en gage, en effectue l'ouverture.
    Renvoie une "preuve" que la mise en gage a été effectuée correctement. La
    preuve contient la valeur de x[i].
    """
    n = len(x)
    proof = []
    if n > 1:
        _, node_tree = secret[0]
        for j in range((n-1).bit_length()-1, -1, -1):
            bit = (i >> j) & 1
            witness_hash, _ = node_tree[1-bit]
            proof.append(witness_hash)
            _, node_tree = node_tree[bit]
    return (base64.b64encode(x[i]).decode(), 
        base64.b64encode(b''.join(reversed(proof))).decode())


if __name__ == "__main__":
    main()