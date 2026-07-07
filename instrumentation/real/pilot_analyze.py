import os, re, collections
d = os.path.join(os.path.dirname(__file__), "poc.trace")
ev = [l.split()[0] for l in open(d) if not l.startswith("#")]
hdr = [l for l in open(d) if l.startswith("# label")][0]
m = re.search(r"root_cause=(\d+) cause=(\d+)", hdr)
assert m, "trace is not labeled anomalous"
rc, ca = int(m[1]), int(m[2])
names = collections.Counter(ev)
freeish = {k: v for k, v in names.items() if re.search(r"free|dealloc|sweep|gc_", k)}
print("free/gc-shaped fns:", freeish)
def rank(pred):
    cands = [i for i in range(len(ev)) if i != rc]
    r = sorted(cands, key=lambda i: (not pred(ev[i]), -i))
    return r.index(ca) + 1
print("positional recency rank:", rank(lambda f: True))
print("most-recent 'free' substring rank:", rank(lambda f: "free" in f))
print("vocab size:", len(names), "| total events:", len(ev))
