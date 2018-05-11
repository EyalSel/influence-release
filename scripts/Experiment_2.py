import numpy as np
from data_poisoning import data_poisoning
from utils import dataset_metadatas, experiment_result_metadata_to_FN, FN_to_experiment_result_metadata, get_dataset, get_full_model_graph

"""
##Experiment2: Targeted vs untargeted
### Knobs:
  - Targeted, multiclass
  - Test points: 7 of Eagle, 7 of Mushroom, 7 of Snail
  - training points: 1
  - Datasets: [Eagle-Mushroom-Snail]

### Metrics:
  - Logits
"""

def run_exp_2():
	for dataset_classes in [dataset_metadatas["Eagle-Mushroom-Snail"]]:
		cache = {}
		cache['data_sets'] = get_dataset(dataset_classes)
		cache['full_model_and_graph'] = get_full_model_graph(dataset_classes, cache['data_sets'])

		#sampled_test_points = np.zeros(2+6+6)
		#sampled_test_points[:2] = np.where(cache['data_sets'].test.labels == 0)[0][:2]
		#sampled_test_points[2:8] = np.where(cache['data_sets'].test.labels == 1)[0][:6]
		#sampled_test_points[8:14] = np.where(cache['data_sets'].test.labels == 2)[0][:6]
		sampled_test_points = np.zeros(7)
		sampled_test_points[:2] = np.where(cache['data_sets'].test.labels == 0)[0][:2]
		sampled_test_points[2:] = np.where(cache['data_sets'].test.labels == 1)[0][:5]
		sampled_test_points = np.asarray(list(map(int, sampled_test_points)))
		num_train_points = 1
		for use_IF in [False]:
			for test_idx in sampled_test_points:
				test_label = cache['data_sets'].test.labels[int(test_idx)]
				target_label_1 = int((test_label + 1)%3)
				target_label_2 = int((test_label - 1)%3)
				imgs_indices_to_poison_1, poisoned_images_1 = data_poisoning(dataset_metadata = dataset_classes, 
																		 use_IF = use_IF,
																		 target_test_idx = [test_idx],
																		 num_to_perterb = num_train_points,
																		 target_labels = [target_label_1],
																		 cache = cache)
				imgs_indices_to_poison_2, poisoned_images_2 = data_poisoning(dataset_metadata = dataset_classes, 
																		 use_IF = use_IF,
																		 target_test_idx = [test_idx],
																		 num_to_perterb = num_train_points,
																		 target_labels = [target_label_2],
																		 cache = cache)
				path_to_save = "Experiment_results/Experiment_2_redo/"
				if use_IF: 
					path_to_save += ""#"IF/"
				else:
					path_to_save += ""#"FC/"
				# placeing target_label in num_poisoned_training_points
				experiment_result_metadata = {
					"Experiment_number":2,
					"contents_type":"indices",
					"method":"IF",
					"dataset_name":dataset_classes["name"],
					"num_poisoned_training_points":target_label_1,
					"test_idx":test_idx
				}
				if not use_IF: 
					experiment_result_metadata["method"] = "FC"
				FN = experiment_result_metadata_to_FN(experiment_result_metadata)
				np.save(path_to_save+experiment_result_metadata_to_FN(experiment_result_metadata), imgs_indices_to_poison_1)
				experiment_result_metadata["contents_type"]="poisons"
				np.save(path_to_save+experiment_result_metadata_to_FN(experiment_result_metadata), poisoned_images_1)       
				
				experiment_result_metadata["num_poisoned_training_points"] = target_label_2
				experiment_result_metadata["contents_type"]="indices"
				FN = experiment_result_metadata_to_FN(experiment_result_metadata)
				np.save(path_to_save+experiment_result_metadata_to_FN(experiment_result_metadata), imgs_indices_to_poison_2)
				experiment_result_metadata["contents_type"]="poisons"
				np.save(path_to_save+experiment_result_metadata_to_FN(experiment_result_metadata), poisoned_images_2)   
              

if __name__ == "__main__":
    np.random.seed(0)
    run_exp_2()