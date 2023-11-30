import os
import time
import numpy as np
from ase.io import read, write
from dscribe.descriptors import SOAP


def obtain_xyz(file):
    lines = open(file, "r").readlines()
    data = []
    for line in lines:
        if len(line.strip().split()) == 8:
            data.append([eval(item) for item in line.strip().split()])
    return np.array(data)

def extract_soap_features(data_file, avg_type="inner"):
    test_file = data_file
    ase_sys = read(test_file, index=':5', format="lammps-data", style="atomic")
    positions = ase_sys[0].get_positions()
    # print(positions.shape)
    # print(ase_sys[0].get_positions())
    soap_desc = SOAP(species=[1, 2, 3, 4], r_cut=rcut, n_max=nmax, l_max=lmax, compression={"mode":"crossover"}, average=avg_type)
    # Create descriptors as numpy arrays or sparse arrays
    soap = soap_desc.create(ase_sys)
    # print(soap)
    return list(soap), positions.reshape(-1).tolist()


def get_graph_features(data_file, one_cut):
    # get the graph data from data_file
    # return the graph data
    test_file = data_file
    ase_sys = read(test_file, index=':5', format="lammps-data", style="atomic")
    positions = ase_sys[0].get_positions()
    types = ase_sys[0].get_chemical_symbols()
    A = []
    E = []
    for i, pos in positions:

    pass

# for training data
avg_type = "inner" # outer
rcut = 6 #4.5
rcut_dict = {
    "random": 6.788,
    "S1300": 6.84,
    "S1100": 6.84,
    "S900": 6.825,
    "S700": 6.81,
    "S500": 6.81,
    "S300": 6.80,
    "S77": 6.79,
    "b2": 6.689
}
nmax = 5
lmax = 5
s = time.time()
situation_type_list = ["77K", "150K", "300K", "500K", "700K", "900K", "1300K"]
# situation_type_list = ["77K", "150K", "300K", "500K", "700K"]
# situation_type_list = ["1300K"]
# test_sample_type_list = ["b2", "random", "S77", "S300", "S500", "S700", "S900", "S1100", "S1300"]
test_sample_type_list = ["b2", "random", "S300", "S500", "S900", "S1100"]
project_dir=r'/Users/shaowei/Desktop/Codes4Shihuama/ML4SRO/'

feature_type = "position"
all_input_features = []
all_labels = []
all_feature_npy_file = os.path.join(project_dir, "features", "train/{}".format(feature_type), "{}_{}_{}_{}_features.npy".format(rcut, nmax, lmax, "all"))
all_label_npy_file = os.path.join(project_dir, "features", "train/{}".format(feature_type), "{}_{}_{}_{}_labels.npy".format(rcut, nmax, lmax, "all"))

for situation_type in situation_type_list:

    input_features = []
    labels = []
    feature_npy_file = os.path.join(project_dir, "features", "train/{}".format(feature_type), "{}_{}_{}_{}_features.npy".format(rcut, nmax, lmax, situation_type))
    label_npy_file = os.path.join(project_dir, "features", "train/{}".format(feature_type), "{}_{}_{}_{}_labels.npy".format(rcut, nmax, lmax, situation_type))

    target_dir = os.path.join(project_dir, "data", situation_type)
    assert os.path.exists(target_dir), "Target dir {} does not exist!".format(target_dir)

    for test_sample_type in test_sample_type_list:
        r_cut = rcut_dict[test_sample_type]
        print("Start {} in {}".format(test_sample_type, situation_type))

        test_sample_dir = os.path.join(target_dir, test_sample_type)

        assert os.path.exists(test_sample_dir), "Test sample dir {} does not exist!".format(test_sample_dir)
        thermal_conductivity_file = os.path.join(test_sample_dir, "thermal_conductivity.txt")
        assert os.path.exists(thermal_conductivity_file), "Thermal conductivity file {} does not exist!".format(thermal_conductivity_file)

        # assert len(th_c_vlaues) == 10, "Thermal conductivity values should be 10, but got {}".format(len(th_c_vlaues))

        # get sopa features for test samples
        # SXX and b2, whose data files are in model
        sample_model_data_file_path_template = ""
        if ("S" in test_sample_type or "b2" in test_sample_type) or situation_type in ["77K", "150K"]:

            model_data_dir = os.path.join(project_dir, "data/model")
            assert os.path.exists(model_data_dir), "Model data dir {} does not exist!".format(model_data_dir)

            sample_model_data_file_path_template = os.path.join(model_data_dir, 
                    "{}K_{}.data".format(test_sample_type.replace("S", ""), "system_id") if "S" in test_sample_type else
                    "{}_{}.data".format(test_sample_type, "system_id"))
        else:
            sample_model_data_file_path_template = os.path.join(test_sample_dir, "system_id", "random.data")

        for line in open(thermal_conductivity_file, "r").readlines():
            system_id, thermal_conductivity = line.strip().split()
            label = eval(thermal_conductivity)
            sys_file = sample_model_data_file_path_template.replace("system_id", str(system_id))

            assert os.path.exists(sys_file), "System file {} does not exist!".format(sys_file)
            sro_features, pos_vector = extract_soap_features(sys_file, avg_type=avg_type)
            features = sro_features if feature_type == "sro" else pos_vector
            features += [eval(situation_type.replace("K", ""))]
            input_features.append(features)
            labels.append(label)

        all_input_features += input_features
        all_labels += labels

        print("Finish {} in {}".format(test_sample_type, situation_type))


    input_features = np.array(input_features)
    labels = np.array(labels)
    os.system("mkdir -p ../data/{}".format(feature_type))
    np.save(feature_npy_file, input_features)
    np.save(label_npy_file, labels)

all_input_features = np.array(all_input_features)
all_labels = np.array(all_labels)
os.system("mkdir -p ../data/{}".format(feature_type))
np.save(all_feature_npy_file, all_input_features)
np.save(all_label_npy_file, all_labels)

e = time.time()
print("Cost {0} mins to generate data".format((e-s)/60))


# for test data

situation_type_list = ["77K", "150K"]
all_input_features = []
all_labels = []
all_feature_npy_file = os.path.join(project_dir, "features", "test/{}".format(feature_type), "{}_{}_{}_{}_features.npy".format(rcut, nmax, lmax, "all"))
all_label_npy_file = os.path.join(project_dir, "features", "test/{}".format(feature_type), "{}_{}_{}_{}_labels.npy".format(rcut, nmax, lmax, "all"))
for situation_type in situation_type_list:
    input_features = []
    labels = []
    feature_npy_file = os.path.join(project_dir, "features", "test/{}".format(feature_type), "{}_{}_{}_{}_features.npy".format(rcut, nmax, lmax, situation_type))
    label_npy_file = os.path.join(project_dir, "features", "test/{}".format(feature_type), "{}_{}_{}_{}_labels.npy".format(rcut, nmax, lmax, situation_type))

    target_dir = os.path.join(r'/Users/shaowei/Desktop/Codes4Shihuama/ML4SRO/related/Post_NEMD/test-nedm/size3', situation_type)
    assert os.path.exists(target_dir), "Target dir {} does not exist!".format(target_dir)

    thermal_conductivity_file = os.path.join(target_dir, "thermal_conductivity.txt")
    assert os.path.exists(thermal_conductivity_file), "Thermal conductivity file {} does not exist!".format(thermal_conductivity_file)

    for line in open(thermal_conductivity_file, "r").readlines():
        system_id, thermal_conductivity = line.strip().split()
        label = eval(thermal_conductivity)
        sys_file = os.path.join(target_dir, system_id, "model.data")

        assert os.path.exists(sys_file), "System file {} does not exist!".format(sys_file)
        sro_features, pos_vector = extract_soap_features(sys_file, avg_type=avg_type)
        features = sro_features if feature_type == "sro" else pos_vector
        features += [eval(situation_type.replace("K", ""))]
        input_features.append(features)
        labels.append(label)
    
    all_input_features += input_features
    all_labels += labels

    input_features = np.array(input_features)
    labels = np.array(labels)
    os.system("mkdir -p ../features/test/{}".format(feature_type))
    np.save(feature_npy_file, input_features)
    np.save(label_npy_file, labels)

all_input_features = np.array(all_input_features)
all_labels = np.array(all_labels)
os.system("mkdir -p ../features/test/{}".format(feature_type))
np.save(all_feature_npy_file, all_input_features)
np.save(all_label_npy_file, all_labels)
