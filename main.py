import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format
import caffe
import dd_functions as dd
import os

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
caffe.set_mode_gpu()
caffe.set_device(0) # select GPU device if multiple devices exist

# Load CNN
print("Loading model")
model_path = '../../datasets/instaBarcelona/models/saved/' # substitute your path here
net_fn   = '../../caffe-master/models/bvlc_googlenet/' + 'deploy.prototxt'
param_fn = model_path + 'instaBCN_Inception_frozen_word2vec_tfidf_iter_120000.caffemodel' #instaBCN_Inception_frozen_word2vec_tfidf_iter_120000 #SocialMedia_Inception_all_glove_tfidf_fromScratch_iter_375000
# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))
net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB


# Load starting image
frame = np.float32(PIL.Image.open('sky1024px.jpg'))

# Random noise image
frame = np.random.rand(575,1024,3) * 255

# print(net.blobs.keys())

s = 0.05 # scale coefficient
octave_n = 4
octave_scale = 1.4
iters = 500
layers = ['inception_4c/output','inception_4d/output','inception_4e/output','inception_5a/output','inception_5b/output','inception_5b/5x5_reduce']
layers = ['inception_4c/output', 'inception_4d/1x1', 'inception_4d/3x3_reduce',
     'inception_4d/3x3', 'inception_4d/5x5_reduce', 'inception_4d/5x5', 'inception_4d/pool', 'inception_4d/pool_proj',
     'inception_4d/output','inception_4e/1x1', 'inception_4e/3x3_reduce',
     'inception_4e/3x3', 'inception_4e/5x5_reduce', 'inception_4e/5x5', 'inception_4e/pool', 'inception_4e/pool_proj',
     'inception_4e/output', 'pool4/3x3_s2', 'inception_5a/1x1', 'inception_5a/3x3_reduce', 'inception_5a/3x3',
     'inception_5a/5x5_reduce', 'inception_5a/5x5', 'inception_5a/pool', 'inception_5a/pool_proj',
     'inception_5a/output', 'inception_5b/1x1', 'inception_5b/3x3_reduce',
     'inception_5b/3x3', 'inception_5b/5x5_reduce', 'inception_5b/5x5', 'inception_5b/pool', 'inception_5b/pool_proj',
     'inception_5b/output']

for layer in layers:
    print(layer)
    directory = '../../datasets/deepdream/InstaBCN_randomNoise/'
    folder_name = layer.replace('/','_') + '_s' + str(s) + '_octn' + str(octave_n) + '_octs' + str(octave_scale)
    if not os.path.exists(directory + folder_name):
        os.makedirs(directory + folder_name)

    h, w = frame.shape[:2]
    frame_i = 0
    # Apply it to its own output
    for i in xrange(iters):
        frame = dd.deepdream(net, frame, octave_n=octave_n, octave_scale=octave_scale, end=layer)
        PIL.Image.fromarray(np.uint8(frame)).save(directory + folder_name + "/%04d.jpg"%frame_i)
        frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
        frame_i += 1
        print("Iteration " + str(frame_i) + " / " + str(iters))

    # Using a guide image
    # guide = np.float32(PIL.Image.open('flowers.jpg'))
    # # showarray(guide)
    # # Pick target layer and extract guide features
    # end = 'inception_3b/output'
    # h, w = guide.shape[:2]
    # src, dst = net.blobs['data'], net.blobs[end]
    # src.reshape(1,3,h,w)
    # src.data[0] = dd.preprocess(net, guide)
    # net.forward(end=end)
    # guide_features = dst.data[0].copy()

    # img_out = dd.deepdream(net, img, end=end, objective=dd.objective_guide, guide_features=guide_features)
    # PIL.Image.fromarray(np.uint8(frame)).save("img_out_guided.jpg")

