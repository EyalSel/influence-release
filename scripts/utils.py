import numpy as np
from skimage import io
from load_animals import *
import tensorflow as tf

from influence.inceptionModel import BinaryInceptionModel

dataset_metadatas = {
    "Dog-Fish":{
        "name": "Dog-Fish",
        "classes":["dog", "fish"],
        "num_train_ex_per_class":900,
        "num_test_ex_per_class":300,
        "batch_size":100
    },
    "Eagle-Mushroom":{
        "name": "Eagle-Mushroom",
        "classes":["Eagle", "Mushroom"],
        "num_train_ex_per_class":500,
        "num_test_ex_per_class":50,
        "batch_size":50
    },
	"Eagle-Mushroom-Snail":{
		"name":"Eagle-Mushroom-Snail",
		"classes":["Eagle", "Mushroom", "Snail"],
		"num_train_ex_per_class":500,
		"num_test_ex_per_class":50,
		"batch_size":50
	}
}

###############################################################################################################################
#### THESE TWO FUNCTIONS COME TOGETHER TO MAP FROM RESULT NAME TO METADATA. WHEN YOU CHANGE ONE YOU MUST CHANGE THE OTHER #####
def experiment_result_metadata_to_FN(metadata):
    return "Experiment_{}_{}_{}_{}_num_train_{}__{}".format(metadata["Experiment_number"], 
                                                            metadata["contents_type"],
                                                            metadata["method"], 
                                                            metadata["dataset_name"], 
                                                            metadata["num_poisoned_training_points"], 
                                                            metadata["test_idx"])

def FN_to_experiment_result_metadata(fn):
    fn = fn[:-4]
    fn = fn.split('/')[-1] # get rid of path leading up to file
    lst = fn.split('_')
    test_idx = int(lst[-1])
    num_poisoned_training_points = int(lst[-3])
    Experiment_number = int(lst[1])
    contents_type = lst[2]
    method = lst[3]
    dataset_name = lst[4]
    return {
        "test_idx":test_idx,
        "num_poisoned_training_points":num_poisoned_training_points,
        "Experiment_number":Experiment_number,
        "contents_type":contents_type,
        "method":method,
        "dataset_name":dataset_name
    }
#### THESE TWO FUNCTIONS COME TOGETHER TO MAP FROM RESULT NAME TO METADATA. WHEN YOU CHANGE ONE YOU MUST CHANGE THE OTHER #####
###############################################################################################################################

def get_dataset(dataset_metadata):
    classes = dataset_metadata["classes"]
    num_train_ex_per_class = dataset_metadata["num_train_ex_per_class"]
    num_test_ex_per_class = dataset_metadata["num_test_ex_per_class"]
    classes_str = '_'.join(classes)
    data_filename = os.path.join('../data/', 'dataset_%s_train-%s_test-%s.npz' % ('-'.join(classes), num_train_ex_per_class, num_test_ex_per_class))
    if not os.path.exists(data_filename):
        extract_and_rename_animals()
    data_sets = load_animals(num_train_ex_per_class=num_train_ex_per_class, 
                             num_test_ex_per_class=num_test_ex_per_class,
                             classes=classes)
    return data_sets

def render_img(img):
    img_copy = np.copy(img)
    img_copy /= 2
    img_copy += 0.5
    io.imshow(img_copy)
    
def get_full_model_graph(datasets_metadata, data_sets, model_type = "inception", use_InceptionResNet=False):
    if model_type == "inception" or "resnet-inception":
        img_side = 299
        num_channels = 3 
        batch_size = 100
        initial_learning_rate = 0.001 
        keep_probs = None
        decay_epochs = [1000, 10000]
        weight_decay = 0.001
    
    training_dataset_classes = datasets_metadata["classes"]
    num_classes = len(training_dataset_classes)
    full_graph = tf.Graph()
    with full_graph.as_default():
        full_model_name = '%s_inception_wd-%s' % ('_'.join(training_dataset_classes), weight_decay)
        full_model = BinaryInceptionModel(
            img_side=img_side,
            num_channels=num_channels,
            weight_decay=weight_decay,
            use_InceptionResNet = use_InceptionResNet,
            num_classes=num_classes, 
            batch_size=batch_size,
            data_sets=data_sets,
            initial_learning_rate=initial_learning_rate,
            keep_probs=keep_probs,
            decay_epochs=decay_epochs,
            mini_batch=True,
            train_dir='output',
            log_dir='log',
            model_name=full_model_name)
    return full_graph, full_model
