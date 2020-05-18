#!/usr/bin/env python3
import datetime
import subprocess
import json
import sys
import textwrap
import tempfile
import base64

VERSION = 2
PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MFYwEAYHKoZIzj0CAQYFK4EEAAoDQgAEV5VrPiQFZw0jfzSNthKSXykClAuMRCKr
VUGvo8B7DfsjOHwB8NmVamu6Tc2jpPs4ADAUfKZhGZPOKVJlKXEvjw==
-----END PUBLIC KEY-----"""
OPENSSL = "openssl"  # à ajuster si la version par défaut a des problèmes

##################################################################

if len(sys.argv) < 2:
    print("USAGE: python3 receipt_checker.py [FILENAME]")
    sys.exit(1)

# charge le reçu
receipt_filename = sys.argv[1]
try:
    with open(receipt_filename) as f:
        serialized_receipt, signature = json.load(f)
        receipt = json.loads(serialized_receipt)
except TypeError:
    print("ERROR")
    sys.exit(2)

if receipt['version'] != VERSION:
    print("Versions non-concordantes (ce programme={}, le reçu={})".format(VERSION, receipt['version']))
    sys.exit(3)

# Vérifie l'authenticité du reçu (ceci fait appel à OpenSSL)
with tempfile.NamedTemporaryFile() as pkey_file, tempfile.NamedTemporaryFile() as sig_file:
    pkey_file.write(PUBLIC_KEY.encode())
    pkey_file.flush()
    sig_file.write(base64.b64decode(signature))
    sig_file.flush()

    args = ['openssl', 'dgst', '-sha256', '-verify', pkey_file.name, '-signature', sig_file.name]
    payload = serialized_receipt.encode()
    result = subprocess.run(args, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if len(result.stderr) > 0:
        print("ÉCHEC DE LA VÉRIFICATION DE LA SIGNATURE DU SERVEUR : ")
        print(result.stderr.decode())
        sys.exit(3)
    if result.stdout.decode() != 'Verified OK\n':
        print("SIGNATURE DU SERVEUR INVALIDE.")
        sys.exit(4)

# Affiche le contenu du reçu
date = datetime.datetime.utcfromtimestamp(receipt['start'])
hard = '\n'.join(textwrap.wrap(receipt['hardware'], subsequent_indent=' '*15, width=65))
soft = '\n'.join(textwrap.wrap(receipt['software'], subsequent_indent=' '*15, width=65))
print("Utilisateurs : {}".format(', '.join(receipt['users'])))
print("Date (début) : {}".format(date))
print("Matrice      : {}".format(receipt['matrix']))
print("Graine       : {}".format(receipt['seed']))
print("Noeuds       : {}".format(receipt['nodes']))
print("Coeurs       : {}".format(receipt['cores']))
print("Durée        : {:.1f}s".format(receipt['running_time']))
print("Hardware     : {}".format(hard))
print("Software     : {}".format(soft))
print("Authenticité : OK (signature du serveur valide).")
print()
