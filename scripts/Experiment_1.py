import numpy as np
from data_poisoning import data_poisoning

"""
## Experiment 1
### Knobs:
  - non-targeted, binary
  - number of test points: 1 x 21
  - number of training points: [1, 2, 4, 8, 16]
  - Datasets: ["Dog/Fish", "Eagle/Mushroom"]

### Metrics:
  - Success rate
  - Angle change
  - Test accuracy change
"""
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


def run_exp_1_IF():
    for dataset_classes in [dataset_metadatas["Dog-Fish"], dataset_metadatas["Eagle-Mushroom"]]:
        sampled_test_points = np.random.choice(np.arange(2*dataset_classes["num_test_ex_per_class"]), 7, replace=False)
        for num_train_points in [1]:
            for test_idx in sampled_test_points:
                imgs_indices_to_poison, poisoned_images = data_poisoning(data_selected = dataset_classes["classes"], 
                                                                         num_train_ex_per_class=dataset_classes["num_train_ex_per_class"], 
                                                                         num_test_ex_per_class=dataset_classes["num_test_ex_per_class"],
                                                                         use_IF = True,
                                                                         target_test_idx = [test_idx],
                                                                         num_to_perterb = num_train_points,
                                                                         target_labels = None)
                path_to_save = "Experiment_results/Experiment_1/"
                experiment_result_metadata = {
                    "Experiment_number":1,
                    "contents_type":"indices",
                    "method":"IF",
                    "dataset_name":dataset_classes["name"],
                    "num_poisoned_training_points":num_train_points,
                    "test_idx":test_idx
                }
                FN = experiment_result_metadata_to_FN(experiment_result_metadata)
                np.save(path_to_save+experiment_result_metadata_to_FN(experiment_result_metadata), imgs_indices_to_poison)
                experiment_result_metadata["contents_type"]="poisons"
                np.save(path_to_save+experiment_result_metadata_to_FN(experiment_result_metadata), poisoned_images)

def run_exp_1_FC():
    for dataset_classes in [dataset_metadatas["Dog-Fish"], dataset_metadatas["Eagle-Mushroom"]]:
        sampled_test_points = np.random.choice(np.arange(2*dataset_classes["num_test_ex_per_class"]), 7, replace=False)
        for num_train_points in [1,2,4,8,16]:
            for test_idx in sampled_test_points:
                imgs_indices_to_poison, poisoned_images = data_poisoning(data_selected = dataset_classes["classes"], 
                                                                         num_train_ex_per_class=dataset_classes["num_train_ex_per_class"], 
                                                                         num_test_ex_per_class=dataset_classes["num_test_ex_per_class"],
                                                                         use_IF = False,
                                                                         target_test_idx = [test_idx],
                                                                         num_to_perterb = num_train_points,
                                                                         target_labels = None)
                path_to_save = "Experiment_results/Experiment_1/"
                experiment_result_metadata = {
                    "Experiment_number":1,
                    "contents_type":"indices",
                    "method":"FC",
                    "dataset_name":dataset_classes["name"],
                    "num_poisoned_training_points":num_train_points,
                    "test_idx":test_idx
                }
                FN = experiment_result_metadata_to_FN(experiment_result_metadata)
                np.save(path_to_save+experiment_result_metadata_to_FN(experiment_result_metadata), imgs_indices_to_poison)
                experiment_result_metadata["contents_type"]="poisons"
                np.save(path_to_save+experiment_result_metadata_to_FN(experiment_result_metadata), poisoned_images)