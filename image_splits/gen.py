with open('../Backup/split.toptee.train.json','rb') as f:
    l_t_r = json.load(f)
with open('new.train.toptee1.json','rb') as f:
    l_t = json.load(f)
l_t_n = l_t_r
l_t_n.extend(l_t)
with open('split.shirt.train.json','wb') as f:
    json.dump(l_s_n,f)
