import os

try:
    from nsml import IS_ON_NSML, DATASET_PATH
except:
    IS_ON_NSML=False

# def test_data_loader(root_path):
#     """
#     Data loader for test data
#     :param root_path: root path of test set.

#     :return: data type to use in user's infer() function
#     """
#     # The loader is only an example, and it does not matter which way you load the data.
#     # loader = np.loadtxt(os.path.join(root_path, 'test', 'test_data'), delimiter=',', dtype=np.float32)
#     # return loader

#     return os.path.join(root_path, 'test')

def feed_infer(output_file, infer_func):
    """
    This is a function that implements a way to write the user's inference result to the output file.
    :param output_file(str): File path to write output (Be sure to write in this location.)
           infer_func(function): The user's infer function bound to 'nsml.bind()'
    """
    if IS_ON_NSML:
        root_path = os.path.join(DATASET_PATH, 'test')
    else:
        root_path = "test"
    
    res = infer_func(root_path)

    if res == None:
        print('res is None')
        return
    
    preds_arr = []

    for i in range(len(res["preds"])):
        pred = res['preds'][i]
        col_str = [res['ids'][i]]
        for col in pred:
            col_str.append(str(col))
        
        preds_arr.append(str(",".join(col_str)))

    with open(output_file, 'w+') as file_writer:
        file_writer.write("\n".join(preds_arr))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')
