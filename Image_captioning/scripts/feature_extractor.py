from swin_transformer_backbone import SwinTransformer as STBackbone
import numpy as np
import os
import json
import argparse
from random import shuffle, seed
import string
from torchvision import transforms as trn
preprocess = trn.Compose([
             trn.Resize([384,384]),
             trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
import torch
import torchvision.models as models
import skimage.io

#-----------model-start---------
Model=STBackbone(
            img_size=384, 
            embed_dim=192, 
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            num_classes=1000
        )
print('load pretrained weights!')
Model.load_weights('./swin_large_patch4_window12_384_22kto1k_no_head.pth')
        # Freeze parameters
for _name, _weight in Model.named_parameters():
     _weight.requires_grad = False
            # print(_name, _weight.requires_grad)
Model.cuda().eval()
#-------------model-end-----------
def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs)
    seed(123) # make reproducible
    dir_att = params['output_dir']+'_att'
    print(os.path.abspath(dir_att))
    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)    
    for i,img in enumerate(imgs):
        # load the image
        I = skimage.io.imread(os.path.join(params['images_root'], img['filepath'], img['filename']))
        # handle grayscale input images
        if len(I.shape) == 2:
            I = I[:,:,np.newaxis]
            I = np.concatenate((I,I,I), axis=2)

        I = I.astype('float32')/255.0
        I = torch.from_numpy(I.transpose([2,0,1])).cuda()
        I = preprocess(I)
        I = I.unsqueeze(0)
        #print(I.shape)
        with torch.no_grad():
            att_feats=Model(I)
            np.savez_compressed(os.path.join( dir_att,str(img['cocoid'])),feat=att_feats.data.cpu().float().numpy())

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
    print('wrote ', params['output_dir'])
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='data', help='output h5 file')

    # options
    parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
