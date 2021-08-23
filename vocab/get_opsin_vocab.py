import glob
import re

# root directory of where opsin is installed
path_to_opsin = sys.argv[1]

pattern = path_to_opsin + "/opsin/opsin-core/src/main/resources/uk/ac/cam/ch/wwmm/opsin/resources/*.xml"
all_tokens = set()
for fn in glob.glob(pattern):
    with open(fn, "r") as f:
        text = f.read()
        stripped = re.sub('(?=<!--)([\s\S]*?)-->', '', text)
        stripped = re.sub('<[^<]+?>', '', stripped)
        for line in stripped.split("\n"):
            line = line.strip()
            tokens = line.split("|")
            all_tokens.update(tokens)

all_tokens.discard("")
with open("opsin_vocab.txt", "w") as f:
    f.write("\n".join(list(all_tokens)))
