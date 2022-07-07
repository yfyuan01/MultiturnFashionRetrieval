# encoding: utf-8
import pickle
import json
# dataset type
# {reference: (image_url, caption, image_id)
# target : (target_url, target_id)}

turn_list = ['three','four','five']
targets = ['train','val']
outputfile = '../data/Multiturn/dress.train.json'
urlfile_name = '/Users/yuanyifei/Downloads/fashion-iq-metadata-master/image_url/asin2url.dress.txt'
url_list = open(urlfile_name,'r').readlines()
url_list = [url.strip('\n').split('\t') for url in url_list]
# url:id
url_dict = {item[1].lstrip(' '):item[0].rstrip(' ') for item in url_list}
outputlist = []
for turns in turn_list:
    for target in targets:
        rawfile_name = '/Users/yuanyifei/Downloads/fashion-iq-master/'+turns+'_hop_new1_'+target
        file_name = '/Users/yuanyifei/GUI/data_'+turns+'_'+target+'.pickle'
        capfile_name = '/Users/yuanyifei/GUI/data_'+turns+'_'+target+'_caption.pickle'
        rawfile = open(rawfile_name,'r').readlines()
        rawfile = ''.join(rawfile).split('\n\n')
        rawfile = rawfile[:len(rawfile) - 1]
        items = [item.split('\n') for item in rawfile]
        a = items[0]
        items = [item[1:] for item in items if item[0] == '']
        items.insert(0, a)
        print items[:10]
        with open(file_name,'rb') as f:
            label = pickle.load(f)
        with open(capfile_name,'rb') as f:
            cap = pickle.load(f)
        print len(label)
        print len(cap)
        print len(items)
        for i in range(len(label)):
            if label[i]=='Yes':
                captions = cap[i]
                item = items[i]
                images = item[::2]
                images = [image.lstrip(' ').rstrip(' ') for image in images]
                reference = []
                reference_dict = {}
                for j in range(len(captions)):
                    reference.append((images[j],captions[j].split('\t'),url_dict[images[j]]))
                reference_dict['reference'] = reference
                print url_dict[images[-1]]+'.jpg'
                reference_dict['target'] = (images[-1],url_dict[images[-1]])
                outputlist.append(reference_dict)
print len(outputlist)
print outputlist[0]
with open(outputfile,'wb') as f:
    json.dump(outputlist,f)









        # print len(true_label)
        # print len(label)
        # print len(cap)
        # print len(items)
        #


