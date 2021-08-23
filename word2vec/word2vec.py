import fasttext
import sys
import os
from scipy.spatial import KDTree
import numpy as np

# usage: either python word2vec.py model_fn
#            or python word2vec.py dataset_fn output_model_fn

#for lr in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
if not os.path.exists(sys.argv[1]):
    raise ValueError("unknown file {}".format(model_fn))

try:
    model = fasttext.load_model(sys.argv[1])
except ValueError as e:
    if len(sys.argv) != 3:
        raise ValueError("must supply model output file")
    if os.path.exists(sys.argv[2]):
        raise ValueError("won't overwrite {}".format(sys.argv[2]))
    lr = 0.0005
    print("LR: {}".format(lr))
    model = fasttext.train_unsupervised(sys.argv[1], model='skipgram', lr=lr, epoch=1, verbose=2, dim=300)
    model.save_model(sys.argv[2])

#print(dir(model))

vocab = model.words
vecs = model.get_output_matrix()

if not os.path.exists("word2vec.big.vecs.npz"):
    np.savez("word2vec.big.vecs.npz", words=vocab, vecs=vecs)

word2i = {vocab[i]: i for i in range(len(vocab))}
vecs = np.array([model[word] for word in vocab])
kdtree = KDTree(vecs)

"""
iran57 = 3, saturated (iridin57)
irin57 = 3, unsaturated (iren57)
etan57 = 4, saturated (etidin57)
et57   = 4, unsaturated
olan57 = 5, saturated (olidin57)
ol57   = 5, unsaturated
inan57 = 6, saturated
inin57 = 6, unsaturated
epan57 = 7, saturated
epin57 = 7, unsaturated
ocan57 = 8, saturated
ocin57 = 8, unsaturated
onan57 = 9, saturated
onin57 = 9, unsaturated

* -imide :: acid anhydride ~ -amide :: -oic acid (1 or 2 acyl group bonded to oxygen or nitrogen )
* -ole :: -ine ~ -olane  :: -inane  or  * -ole :: -olane ~ -ine :: -inane
"""

rings = [
         [("iran57", "iridin57"), ("irin57", "iren57")],
         [("etan57", "etidin57"), ("et57",)],
         [("olan57", "olidin57"), ("ol57",)],
         [("inan57",), ("inin57",)],
         [("epan57",), ("epin57",)],
         [("ocan57",), ("ocin57",)],
         [("onan57",), ("onin57",)]
        ]

#A :: B ~ C :: D (A - B + D ~ C)
#A :: C ~ B :: D (A - C + D ~ B)

# a :: b ~ c :: d
def check_analogy(a, b, c, d):
    query_vec = vecs[word2i[a]] - vecs[word2i[b]] + vecs[word2i[d]]
    dists, nns = kdtree.query(query_vec, k=10)
    #print("(hoping {} - {} + {} ~ {})".format(a, b, d, c))
    for i, (dist, idx) in enumerate(zip(dists, nns)):
        # don't print locants/punctuation/etc. unless the query is one
        if len(vocab[idx]) > 3 or len(a) <= 3:
            if vocab[idx][-2:] == "57" and vocab[idx] not in [a, b, d]:
                if vocab[idx] == c:
                    print("(hoping {} - {} + {} ~ {})".format(a, b, d, c))
                    print("({}) {}: {}".format(i, vocab[idx], dist))
    print()

"""
for i in range(len(rings)):
    if len(rings[i][0]) == 2 and len(rings[i][1]) == 2:
        # rings[i][0][0] and rings[i][0][1] mean (I think?) the same thing, so
        # this analogy should hold
        check_analogy(rings[i][0][0], rings[i][0][1], rings[i][1][0], rings[i][1][1])
    for j in range(i + 1, len(rings)):
        # I think these are the same, but do both anyway
        # i-ring saturated :: i-ring unsaturated ~ j-ring saturated :: j-ring unsaturated
        # i-ring saturated :: j-ring saturated ~ i-ring unsaturated :: j-ring unsaturated
        for isat in rings[i][0]:
            for iunsat in rings[i][1]:
                for jsat in rings[j][0]:
                    for junsat in rings[j][1]:
                        check_analogy(isat, iunsat, jsat, junsat)
                        check_analogy(isat, jsat, iunsat, junsat)
exit()
"""
# for each query, q[0] is to q[1] as q[2] is to q[3]
# i.e. q[0] - q[1] + q[3] should be close to q[2]
queries = [("prop61", "pent61", "but61", "hex61"),
           ("phenyldf", "benzen7a", "hydroxy78", "ol57"),
           #("naphthalen7a", "benzen7a", "biphenylen7a", "phenyldf"),
           ("naphthalen7a", "benzen7a", "eth61", "meth61"),
           #("tri72", "monod5", "tetr6d", "di72"),
           ("hex6d", "tetr6d", "dec6d", "oct6d"),
           ("oct6d", "tetr6d", "dec6d", "pent6d"),
           ("oct6d", "tetr6d", "dodec6d", "hex6d"),
           ("dec6d", "pent6d", "dodec6d", "hex6d"),
           ("[6f", "]63", "(54", ")55"),
           ("imine73", "al73", "amine73", "ol73"),
           ("phosphonous_acid00", "nitrous_acid00", "phosphoroso00", "nitroso00"),
           ("diphosphate00", "disulfate00", "phosphate00", "sulfate00"),
           ("selenate00", "tellurate00", "selenite00", "tellurite00"),
           ("xanthen00", "thioxanthen00", "phenoxazin00", "phenothiazin00"),
           #("pyrano00", "thiopyrano00", "chromeno00", "thiochromeno00")
           ("chromen00", "quinolin00", "isochromen00", "isoquinolin00")
           ]
#monfa
#* -imide :: acid anhydride ~ -amide :: -oic acid (1 or 2 acyl group bonded to oxygen or nitrogen )
#* -ole :: -ine ~ -olane  :: -inane  or  * -ole :: -olane ~ -ine :: -inane

for query in queries:
    query = [q[:-2] for q in query]
    query_vec = vecs[word2i[query[0]]] - vecs[word2i[query[1]]] + vecs[word2i[query[3]]]

    dists, nns = kdtree.query(query_vec, k=10)
    print("(hoping {} - {} + {} ~ {})".format(query[0], query[1], query[3], query[2]))
    print("*{}: {}".format(query[2], np.linalg.norm(query_vec - vecs[word2i[query[2]]])))
    r = np.random.randint(len(vocab))
    print("X{}: {}".format(vocab[r], np.linalg.norm(query_vec - vecs[r])))

    for dist, idx in zip(dists, nns):
        if len(vocab[idx]) > 3 or len(query[0]) == 3 or True:
            print("{}: {}".format(vocab[idx], dist))
        #print(vocab[nn], (vecs[nn] * vecs[query_i]) / (vecs[nn].norm() * vecs[query_i].norm()))
    print()
