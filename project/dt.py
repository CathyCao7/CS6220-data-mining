import DecisionTree
import re
import sys
import operator

debug = 0
def convert(value):
  try:
        answer = float(value)
        return answer
  except:
        return value

def sample_index(sample_name):
    m = re.search('_(.+)$', sample_name)
    return int(m.group(1))

def get_test_data_from_csv():
    global all_class_names, feature_values_for_samples_dict, class_for_sample_dict
    if not test_datafile.endswith('.csv'): 
        sys.exit("Aborted. get_test_data_from_csv() is only for CSV files")
    class_name_in_column = 49 - 1 
    all_data = [line.rstrip().split(',') for line in open(test_datafile,"rU")]
    data_dict = {line[0] : line[1:] for line in all_data}
    if 'encounter_id' not in data_dict:
        sys.exit('''Aborted. The first row of CSV file must begin '''
                 '''with "" and then list the feature names and the class names''')
    feature_names = [item.strip('"') for item in data_dict['encounter_id']]
    class_column_heading = feature_names[class_name_in_column]
    feature_names = [feature_names[i-1] for i in training_file_columns_for_feature_values]
    class_for_sample_dict = { "sample_" + key.strip('"') : class_column_heading + "=" +
                              data_dict[key][class_name_in_column] for key in data_dict if key != 'encounter_id'}
    feature_values_for_samples_dict = {"sample_" + key.strip('"') : 
          list(map(operator.add, list(map(operator.add, feature_names, "=" * len(feature_names))),
                [str(convert(data_dict[key][i-1].strip('"'))) 
                    for i in training_file_columns_for_feature_values])) for key in data_dict if key != 'encounter_id'}
    features_and_values_dict = {data_dict['encounter_id'][i-1].strip('"') : 
            [convert(data_dict[key][i-1].strip('"')) for key in data_dict if key != 'encounter_id'] 
                                                         for i in training_file_columns_for_feature_values} 
    all_class_names = sorted(list(set(class_for_sample_dict.values())))
    numeric_features_valuerange_dict = {}
    feature_values_how_many_uniques_dict = {}
    features_and_unique_values_dict = {}
    for feature in features_and_values_dict:
        unique_values_for_feature = list(set(features_and_values_dict[feature]))
        unique_values_for_feature = sorted(list(filter(lambda x: x != 'NA', unique_values_for_feature)))
        feature_values_how_many_uniques_dict[feature] = len(unique_values_for_feature)
        if all(isinstance(x,float) for x in unique_values_for_feature):
            numeric_features_valuerange_dict[feature] = [min(unique_values_for_feature), max(unique_values_for_feature)]
            unique_values_for_feature.sort(key=float)
        features_and_unique_values_dict[feature] = sorted(unique_values_for_feature)
    if debug:
        print("\nAll class names: " + str(all_class_names))
        print("\nEach sample data record:")
        for item in sorted(feature_values_for_samples_dict.items(), key = lambda x: sample_index(x[0]) ):
            print(item[0]  + "  =>  "  + str(item[1]))
        print("\nclass label for each data sample:")
        for item in sorted(class_for_sample_dict.items(), key=lambda x: sample_index(x[0])):
            print(item[0]  + "  =>  "  + str(item[1]))
        print("\nfeatures and the values taken by them:")
        for item in sorted(features_and_values_dict.items()):
            print(item[0]  + "  =>  "  + str(item[1]))
        print("\nnumeric features and their ranges:")
        for item in sorted(numeric_features_valuerange_dict.items()):
            print(item[0]  + "  =>  "  + str(item[1]))
        print("\nnumber of unique values in each feature:")
        for item in sorted(feature_values_how_many_uniques_dict.items()):
            print(item[0]  + "  =>  "  + str(item[1]))

training_datafile = "bag11.csv"
test_datafile = 'bag1.csv'
training_file_columns_for_feature_values = [2,3,4,18,19,20,41,47,48]
dt = DecisionTree.DecisionTree(
                training_datafile = training_datafile,
                csv_class_column_index = 49,
                csv_columns_for_features = [2,3,4,18,19,20,41,47,48],
                entropy_threshold = 0.01,
                max_depth_desired = 5,
                symbolic_to_numeric_cardinality_threshold = 10,
     )

dt.get_training_data()
dt.calculate_first_order_probabilities()
dt.calculate_class_priors()
dt.show_training_data()
root_node = dt.construct_decision_tree_classifier()
root_node.display_decision_tree("   ")

get_test_data_from_csv()

FILEOUT   = open('out.csv', 'w')

class_names = "readmitted"
output_string = "encounter_id," + class_names + "\n"
FILEOUT.write(output_string)
listclassification = []
for item in sorted(feature_values_for_samples_dict.items(), key = lambda x: sample_index(x[0]) ):
    test_sample =  feature_values_for_samples_dict[item[0]]
    classification = dt.classify(root_node, test_sample)
    del classification['solution_path']
    which_classes = list( classification.keys() )
    which_classes = sorted(which_classes, key=lambda x: classification[x], reverse=True)
    chooseclass = which_classes[0]
    listclassification.append(chooseclass)
    output_string = str(sample_index(item[0]))
    output_string +=  "," + chooseclass[11:]
    FILEOUT.write(output_string + "\n")
FILEOUT.close()

listtrueclass = []
for item in sorted(class_for_sample_dict.items(), key=lambda x: sample_index(x[0])):
    trueclass = item[1]
    listtrueclass.append(trueclass)
print len(listclassification)
count = 0
for i in range(len(listclassification)):
    if listclassification[i] == listtrueclass[i]:
        count = count + 1
print count
accuracy = float(count) / len(listclassification)
print accuracy


