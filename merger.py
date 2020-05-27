import os
import glob
import h5py

DATA_ITEMS = [u'activations', u'file_names', u'label_mappings', u'labels']


def merge(input_filenames, output_filename):
    ''' create a single h5 file from several input ones.
    '''
    output_file = h5py.File(output_filename)
    for input_filename in input_filenames:
        print(input_filename)
        group_name = os.path.basename(input_filename).split('_')[0]
        group = output_file.create_group(group_name)
        input_file = h5py.File(input_filename, 'r')
        for key in DATA_ITEMS:
            group.create_dataset(key, data=input_file[key])
        for key in input_file[u'activations'].attrs:
            group[u'activations'].attrs[key] = input_file[u'activations'].attrs[key]
        # activation_labels is an odd one
        label_group = group.create_group('activation_labels')
        # There _should_ only be a single field value for this file.
        # There _should_ only be a single field value for this file
        # assert len(input_file['activation_labels']) == 1
        field_name = [x for x in input_file['activation_labels']][0]
        label_group.create_dataset(group_name, data=input_file['activation_labels/{}'.format(field_name)])
        input_file.close()
    output_file.close()


def merge_layer(directory, layer_name, name_leader='AN_merged_all', suffix='', input_filenames='n*_fc8_max.h5',
                isimagenet=True):
    ''' look in directory for all h5s for a layer and merge them. '''
    # input_filenames = glob.glob(os.path.join(directory, name_leader+'n*_{}.h5'.format(layer_name)))
    input_filenames = glob.glob(os.path.join(directory, input_filenames))
    # output_filename = os.path.join(directory, name_leader+'merged_{}.h5'.format(layer_name))
    # input_filenames = glob.glob(os.path.join(directory, 'n*_{}{}.h5'.format(layer_name, suffix)))
    output_filename = os.path.join(directory, '{}_{}{}.h5'.format(name_leader, layer_name, suffix))
    print(input_filenames)
    merge(input_filenames, output_filename)
    return output_filename


def main():
    # merge(['n02096051_fc6.h5','n02129604_fc6.h5','n02640242_fc6.h5'], 'fc6.h5')
    # merge_layer(os.getcwd(), 'prob')
    # merge_layer(os.getcwd(), 'fc7')
    merge_layer(os.getcwd(), 'fc8', '', 'all')
    # merge_layer(os.getcwd(), 'conv5')
    # merge_layer(os.getcwd(), 'fc6')


if __name__ == '__main__':
    main()
