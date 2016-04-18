from numpy import *
import os.path
from numpy.lib.recfunctions import *

def parse_pair_filename(filename):
    tdc, rung, grouping, pair_id = filename.split(".")[0].split("_")
    tdc = int(tdc[len('tdc'):])
    rung = int(rung[len('rung'):])
    pair_id = pair_id[len('pair'):]
    part = 0.0
    if 'A' in pair_id or 'B' in pair_id:
        part = {'A':0.0, 'B': 0.5}[pair_id[-1]]
        pair_id = pair_id[:-1]
    pair_id = float(pair_id) + part
    return tdc, rung, pair_id

pairs = None
for dirpath, dirnames, filenames in os.walk(os.path.join(os.getcwd(), 'tdc1')):
    for filename in filenames:
        if not filename.endswith('.txt'): continue
        filepath = os.path.join(dirpath, filename)
        if 'truth' not in filename:
            tdc, rung, pair_id = parse_pair_filename(filename)            
            pair_data = loadtxt(filepath, skiprows=6, 
                              dtype={"names":("time", "lcA", "errA", "lcB", "errB"),
                                     "formats":("f4", "f4", "f4", "f4", "f4")})
            pair_data = append_fields(pair_data, 'tdc', [], dtypes='<f8')
            pair_data = append_fields(pair_data, 'rung', [], dtypes='<f8')
            pair_data = append_fields(pair_data, 'pair', [], dtypes='<f8')
            pair_data = pair_data.filled()
            pair_data['tdc'] = tdc
            pair_data['rung'] = rung
            pair_data['pair'] = pair_id
            
            if pairs is not None:
                pairs = append(pairs, pair_data)
            else:
                pairs = pair_data

truths = None
for dirpath, dirnames, filenames in os.walk(os.path.join(os.getcwd(), 'tdc1')):
    for filename in filenames:
        if not filename.endswith('.txt'): continue
        filepath = os.path.join(dirpath, filename)
        if 'truth' in filename:
            tdc, rung = dirpath.split("/")[-2:]
            tdc = int(tdc[len("tdc"):])
            rung = int(rung[len("rung"):])
            truth = loadtxt(filepath, skiprows=1,
                    dtype={"names":("pairfile", "dt", "m1", "m2", "zl", "zs", "id", "tau", "sig"),
                           "formats":("S30", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f4")})
            truth = append_fields(truth, 'tdc', [], dtypes='<f8')
            truth = append_fields(truth, 'rung', [], dtypes='<f8')
            truth = append_fields(truth, 'pair', [], dtypes='<f8')
            truth = truth.filled()
            truth['tdc'] = tdc
            truth['rung'] = rung
            for i in xrange(0, len(truth)):
                if 'test' in truth[i]['pairfile']: continue
                tdc, rung, pair_id = parse_pair_filename(truth[i]['pairfile'])
                truth[i]['pair'] = pair_id
            if truths is not None:
                truths = append(truths, truth)
            else:
                truths = truth

savez('pairs.npz', pairs)
savez('truths.npz', truths)


for field in ("dt", "m1", "m2", "zl", "zs", "id", "tau", "sig"):
    pairs = append_fields(pairs, field, [], dtypes='<f4', fill_value=nan)
pairs = pairs.filled()


for i in xrange(0, len(truths)):
    tdc = truths[i]['tdc']
    rung = truths[i]['rung']
    pair = truths[i]['pair']
    
    filt = where((pairs['tdc'] == tdc) & (pairs['rung'] == rung) & (pairs['pair'] == pair))[0]
    
    for field in ("dt", "m1", "m2", "zl", "zs", "id", "tau", "sig"):
        pairs[field][filt] = truths[field][i]


pairs = append_fields(pairs, 'full_pair_id', [], dtypes='<f8')
pairs['full_pair_id'] = pairs['pair'] + pairs['rung'] * 10000 + pairs['tdc'] * 100000

savez('pairs_with_truths.npz', pairs)



