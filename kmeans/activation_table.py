""" activation_table
Provides interfaces to image activation dumps.
"""
import functools
import os
from typing import Tuple, List, KeysView, Union, Dict

import h5py
import numpy as np
from h5py._hl.files import File as h5File

from kmeans.activation import Activation, LabelType
from kmeans.neuron import Neuron

ActivationIndex = Tuple[str, int]
ActivationImplicitIndex = Union[ActivationIndex, int]


def _strip_h5(value: str) -> str:
    """ remove trailing .h5, if present """
    if value.endswith(".h5"):
        value = value[:-3]
    return value


# pylint: disable=too-many-instance-attributes
class ActivationObject(object):
    """ An activation data instance.
    Contains the neuron activation values from inception for an image set

    Doesn't define _where_ the data comes from - that should be implemented by
    child classes

    Attributes:
        index_name (str): a unique (ideally) identifier for this object
    """

    def __init__(self, index_name: str) -> None:
        self.index_name = index_name
        self.image_count = 0
        self.neuron_count = 0
        self.activations = np.array([])
        self.label_mappings = []
        self.labels = np.array([])
        self.activation_labels = {}
        self.file_names = []

    def is_valid(self) -> bool:
        return True

    def get_activation(
            self,
            idx: int,
            standardise: bool = False,
            mean: bool = True) -> Activation:
        """Get a point by it's index.

        Args:
            idx (int): the index (from 0) of the activation to get
            standardise (bool, optional): whether to standardise the activation
                so that the maximum distance in any dimension is 1.0
            mean (bool, optional): for multidimensional data, whether to convert
                it to 1D by averaging the values in dimensions 2+
        """
        if idx >= self.image_count:
            raise ValueError("index {} out of bounds.".format(idx))
        if mean:
            vector = np.mean(np.mean(self.activations[idx], axis=0), axis=0)
        else:
            # Note that reshape forces vector to be 1D!
            vector = self.activations[idx].reshape(-1)

        try:
            labels = self.label_mappings[idx]
        except IndexError as e:
            print("ERROR: Failed to get label mappings for {}".format(idx))
            print("Availiable mappings: {}".format(self.label_mappings))
            raise e
        point = Activation(
            labels=self.label_mappings[idx],
            vector=vector,
            index=(
                self.index_name,
                idx))
        if standardise:
            point.standardise()

        return point

    def get_activations(
            self,
            indices: List[int],
            standardise: bool = False,
            mean: bool = True) -> List[Activation]:
        """Get a list of points, by index.

        Note:
            parameters as per get_activation
        """
        return [self.get_activation(idx, standardise, mean) for idx in indices]

    def get_all_activation_indices(self) -> range:
        """Get indices of  points in the table.
        """
        return range(self.image_count)

    def get_batch_indices(self, batch_size: int, offset: int = 0) -> List[int]:
        """ get a list of indices of batch_size elements of each label
            starting from offset*batch_size
        """
        # TODO: make sure we don't go off the end
        indices = []
        start_offset = batch_size * offset
        # We assume that the label names occur in label order
        for label in self.labels:
            try:
                start_idx = self.activation_labels[str(
                    label)][0] + start_offset
                indices += range(start_idx, start_idx + batch_size)
            except KeyError:
                # Handle empty directories
                pass
        return indices


class ActivationGroup(
        ActivationObject):  # pylint: disable=too-many-instance-attributes
    """ a single activation object loaded from a h5 file
    Contains the neuron activation values for an image set
    """

    def __init__(self, source: h5File, index_name: str) -> None:
        """ Creates the object:
            source: a h5py file or group object
            index_name: the name by which the parent activation table refers to this object.
        """
        ActivationObject.__init__(self, index_name)
        try:
            self.activations = source['activations']
        except ValueError:
            import pdb
            pdb.set_trace()
        self.labels = source['labels']
        self.label_mappings = source['label_mappings']
        self.activation_labels = source['activation_labels']
        try:
            self.file_names = source['file_names']
        except KeyError:
            # hdf5 made with old version
            print("Warning: This hdf5 file lacks file_name mappings")
        self.image_count = self.activations.attrs['image_count']
        # TODO: handle multi-dimensional neurons better
        self.neuron_count = self.activations.attrs['neuron_count']
        self._source = source


class ActivationFile(
        ActivationGroup):  # pylint: disable=too-many-instance-attributes
    """ An activation.hdf5 file
    Contains the neuron activation values from inception for an image set
    """

    def __init__(self, filename: str, index_name: str) -> None:
        file_handle = h5py.File(
            filename, 'r')  # pylint: disable=undefined-variable
        ActivationGroup.__init__(self, file_handle, index_name)


class ActivationTable(object):
    """Activation_table
    Provides a bunch of helper functions to access an activation hdf5 archive.
    """

    def __init__(self, standardise: bool = False, mean: bool = True) -> None:
        self.activation_files = {}
        self.default_file = None
        self.standardise = standardise
        self.mean = mean
        self.merged_file = None
        self.normalised_file = None
        self.normalisation_min = None
        self.normalisation_max = None

    def add_merged_file(self, filename: str) -> None:
        """ Attempts to open an hdf5 of multiple activation_tables and read the values. """
        if self.merged_file is not None or self.normalised_file is not None:
            # Wouldn't be too hard to change this though.
            raise ValueError('Only one merged / normalised file is supported at a time.')
        if self.default_file is not None:
            # Again not that hard to change
            raise ValueError(
                'Cannot mix merged and unmerged files')
        self.merged_file = h5py.File(filename, 'r')
        for merged_key in self.merged_file:
            self.activation_files[merged_key] = ActivationGroup(
                self.merged_file[merged_key], merged_key)
        try:
            self.default_file = list(self.merged_file.keys())[0]
        except IndexError:
            import pdb
            pdb.set_trace()

    def add_normalised_file(self, filename: str) -> None:
        """ Attempts to load an hdf5 file of multiple normalised activations and read the values."""
        if self.merged_file is not None or self.normalised_file is not None:
            # Wouldn't be too hard to change this though.
            raise ValueError('Only one merged / normalised file is supported at a time.')
        if self.default_file is not None:
            # Again not that hard to change
            raise ValueError(
                'Cannot mix merged and unmerged files')
        self.merged_file = h5py.File(filename, 'r')
        self.normalisation_min = self.merged_file['normalisation_min']
        self.normalisation_max = self.merged_file['normalisation_max']

        for merged_key in self.merged_file['values']:
            self.activation_files[merged_key] = ActivationGroup(
                self.merged_file['values'][merged_key], merged_key)
        self.default_file = list(self.merged_file.keys())[0]

    def save_normalised_file(self, filename: str) -> None:
        """ Attempts to save a new hdf5 file containing a normalised version of this file."""
        if self.normalised_file is not None:
            raise ValueError("Can't renormalise a file.")

        # First calculate the value range of our activation files.
        min_values = [np.min(self.activation_files[x].activations, axis=0) for x in self.activation_files]
        max_values = [np.max(self.activation_files[x].activations, axis=0) for x in self.activation_files]

        # Now calculate the value range across them
        self.normalisation_min = functools.reduce(np.minimum, min_values)
        self.normalisation_max = functools.reduce(np.maximum, max_values)
        norm_vector = self.normalisation_max - self.normalisation_min

        with h5py.File(filename, 'w') as output:
            output['normalisation_min'] = self.normalisation_min
            output['normalisation_max'] = self.normalisation_max
            value_grp = output.create_group('values')
            for afile in self.activation_files:
                # try:
                #     src = self.activation_files[afile]
                # except:
                #     src = self.activation_files[afile]
                src = self.activation_files[afile]
                src_file = src._source
                grp = value_grp.create_group(afile)
                for field in src_file:
                    if field == 'activations':
                        grp['activations'] = (src.activations - self.normalisation_min) / norm_vector
                    else:
                        src_file.copy(field, grp)
                grp['activations'].attrs['image_count'] = src.image_count
                grp['activations'].attrs['neuron_count'] = src.neuron_count

    def add_file(self, filename: str) -> None:
        """ Attempts to open the activation hdf5 file and read the basic values """
        base_filename = os.path.basename(filename)
        if base_filename in self.activation_files:
            raise ValueError(
                'hdf5 file {} already loaded.'.format(base_filename))
        self.activation_files[base_filename] = ActivationFile(
            filename, base_filename)
        if self.default_file is None:
            self.default_file = base_filename
        # TODO: maybe validate the file in some way? That the label counts are
        # the same, etc.

    def add_direct(self, identifier: object,
                   image_count: int,
                   neuron_count: int,
                   labels: List[str],
                   neuron_x_count: int,
                   neuron_y_count: int) -> 'ActivationDirect':
        """ creates an activation object which will hold its data in ram.
            takes an identifier to describe the object.
            returns a handle to the object so you can call add_activation
        """
        ident = str(identifier)
        if ident in self.activation_files:
            raise ValueError('identifier {} already in use.'.format(ident))
        self.activation_files[ident] = ActivationDirect(ident,
                                                        image_count,
                                                        neuron_count,
                                                        labels,
                                                        neuron_x_count,
                                                        neuron_y_count)
        if self.default_file is None:
            self.default_file = ident

        return self.activation_files[ident]

    def get_loaded_files(self) -> KeysView:
        """ Return the list of filenames loaded.
        """
        return self.activation_files.keys()

    def get_activation(self, idx: ActivationImplicitIndex) -> Activation:
        """Get a point by index. """
        filename, index = self._expand_index(idx)
        return self.activation_files[filename].get_activation(
            index, self.standardise, self.mean)

    def get_all_point_indices(self) -> List[ActivationIndex]:
        """Get all points in the table.
        """
        points = []
        for act_file in self.activation_files:
            points += [(act_file, x) for x in
                       self.activation_files[act_file].get_all_activation_indices()]
        return points

    def get_file_name(self, index: ActivationIndex) -> str:
        """ Gets the filename associate with a given index.
        """
        act_file, index = self._expand_index(index)

        try:
            return self.activation_files[act_file].file_names[index]
        except KeyError:
            # some clusters ended up up logged using the file extension for some reason.
            return self.activation_files[act_file + ".h5"].file_names[index]

    def get_image_count(self) -> int:
        """ returns the number of images in the table """
        return sum([x.image_count for x in self.activation_files.values()])

    def _global_to_local_index(self, index: int) -> ActivationIndex:
        """ convert the given index into a tuple pair as used internally. """
        for a_file_key in self.activation_files:
            a_file = self.activation_files[a_file_key]
            if index < a_file.image_count:
                return _strip_h5(a_file_key), index
            index -= a_file.image_count
        # should never get here
        raise IndexError('index out of range')

    def _local_to_global_index(self, index: ActivationIndex) -> int:
        """ convert the given index tuple into a global index."""
        local_key_full, local_index = index
        local_key = _strip_h5(local_key_full)
        for a_file_key in self.activation_files:
            a_base_key = _strip_h5(a_file_key)
            if len(local_key) < 9 or len(a_base_key) < 9:
                import pdb
                pdb.set_trace()
            if local_key == a_base_key:
                return local_index
            local_index += self.activation_files[a_file_key].image_count
        # should never get here
        raise KeyError('invalid index tuple.')

    def to_local_indices(self, index_list: List[int]) -> List[ActivationIndex]:
        """ convert a given list of indices into local versions suitible for use in getpoint, etc
        """
        if len(index_list) == 0:
            return []

        image_count = self.get_image_count()
        out_of_range = [x for x in index_list if x > image_count]
        if len(out_of_range) > 0:
            raise IndexError(
                'One or more indices were out of range (max is {})'.format(image_count))

        return [self._global_to_local_index(x) for x in index_list]

    def from_local_indices(
            self,
            index_list: List[ActivationIndex]) -> List[int]:
        """ convert a given list of local tuple indices into normal global version.
        """
        return [self._local_to_global_index(x) for x in index_list]

    def _get_activations_for_neuron_old(self, neuron: int) -> Neuron:
        """ Get a list of all the activations for a given neuron, loading from an old file.
            If your activation tables are not all the same shape, this will probably explode.
        """
        output_shape = list(self.activation_files.values())[
            0].activations.shape[1:-1] + (self.get_image_count(),)
        result = np.zeros(output_shape, dtype='f')
        cur_index = 0
        for a_file in self.activation_files.values():
            try:
                result[:, :, cur_index:cur_index +
                       a_file.image_count] = a_file.activations[:].T[neuron, :, :, :]
                cur_index += a_file.image_count
            except BaseException:
                import pdb
                pdb.set_trace()
        return Neuron(str(neuron), np.squeeze(result))

    def get_activations_for_neuron(self, neuron: int) -> Neuron:
        """ Get a list of all the activations for a given neuron.
            If your activation tables are not all the same shape, this will probably explode.
        """
        # So, the problem is that old activation files are arranged
        # image,x,y,neron, while newer ones are image,neuron,x,y
        if list(self.activation_files.values())[0].activations.shape[-1] > 1:
            print("Loading old seqence neuron")
            return self._get_activations_for_neuron_old(neuron)

        output_shape = list(self.activation_files.values())[
            0].activations.shape[2:] + (self.get_image_count(),)
        result = np.zeros(output_shape, dtype='f')
        cur_index = 0
        for a_file in self.activation_files.values():
            try:
                result[:, :, cur_index:cur_index +
                       a_file.image_count] = a_file.activations[:].T[:, :, neuron, :]
                cur_index += a_file.image_count
            except BaseException:
                import pdb
                pdb.set_trace()
        return Neuron(str(neuron), np.squeeze(result))

    def _expand_index(self, index: ActivationImplicitIndex) -> ActivationIndex:
        """ ensures an index is in the correct tuple format. """
        if isinstance(index, int):
            return self.default_file, index

        return index

    def _expand_indices(
            self,
            indices: List[ActivationImplicitIndex]) -> List[ActivationIndex]:
        """ ensures a list of indices is in the correct tuple format."""
        result = []
        for index in indices:
            try:
                result.append((index[0], index[1]))
            except TypeError:
                result.append((self.default_file, index))
        return result

    def _split_indices_by_file(
            self, indices: List[ActivationIndex]) -> Dict[str, List[int]]:
        """ convert a list of (file,index) tuples into a dictionary of {file}->[indices]
        """
        indices = self._expand_indices(indices)
        point_lists = {}
        for point_file, point_idx in indices:
            try:
                point_lists[point_file].append(point_idx)
            except KeyError:
                point_lists[point_file] = [point_idx]
        return point_lists

    def get_activations(
            self,
            indices: List[ActivationImplicitIndex]) -> List[Activation]:
        """Get a list of points, by index.
           These will either be indices or (filename,index) tuples
        """
        results = []
        point_files = self._split_indices_by_file(indices)
        for point_file in point_files:
            results += self.activation_files[point_file]. get_activations(
                point_files[point_file], self.standardise, self.mean)

        return results

    def get_batch_indices(self, batch_size: int, offset: int = 0) -> List[int]:
        """ get a list of indices of batch_size elements of each label per activation_file
            starting from offset*batch_size
            Offset determines which batch to get.
            e.g. with batch_size of 20 and offset 1,
            it will return the 21st through 40th of each batch
        """
        results = []
        for act_file in self.activation_files:
            results += [(act_file, x) for x in
                        self.activation_files[act_file].get_batch_indices(batch_size, offset)]
        return results


# pylint: disable=too-many-instance-attributes
class ActivationDirect(ActivationObject):
    """ variant of activation table that loads the bottlenecks from memory
    """

    def __init__(self, index_name: str,
                 image_count: int,
                 neuron_count: int,
                 labels: List[str],
                 neuron_x_count: int = 8,
                 neuron_y_count: int = 8) -> None:
        """ constructor.
            image_count: number of images that will be loaded
            neuron_count: number of neurons in the system.
            labels: list of label names
        """
        ActivationObject.__init__(self, index_name)
        self.neuron_count = neuron_count
        self.neuron_x_count = neuron_x_count
        self.neuron_y_count = neuron_y_count
        self.image_count = image_count
        self.activations = np.zeros(
            (image_count,
             neuron_count,
             neuron_x_count,
             neuron_y_count),
            dtype='f')
        self.labels = np.array(labels)
        self._index = 0

    def add_activation(
            self,
            activation_values: np.array,
            file_name: str,
            labels: LabelType) -> None:
        """ Add a single file's activations to the table. """
        try:
            self.activations[self._index, :] = activation_values
        except ValueError:
            if self.neuron_x_count == 1 and self.neuron_y_count == 1:
                try:
                    self.activations[self._index, :] = activation_values.reshape(
                        self.neuron_count, 1, 1)
                except ValueError:
                    import pdb
                    pdb.set_trace()
            else:
                import pdb
                pdb.set_trace()
        self.file_names.append(file_name)
        if isinstance(labels, str):
            labels_list = [labels]
        else:
            labels_list = labels

        self.label_mappings.append(labels_list)

        for label in labels_list:
            try:
                self.activation_labels[label].append(self._index)
            except KeyError:
                self.activation_labels[label] = [self._index]
        self._index += 1

    def save_to_hdf5(
            self,
            file_name: str,
            regenerate_labels: bool = False) -> None:
        """ Freeze this down into a normal hdf5 file.
            optionally generate the label list from the added activations
        """
        if regenerate_labels:
            self.labels = self.activation_labels.keys()

        # prevent h5py string encoding issues
        self.labels = [x.encode('utf8') for x in self.labels]
        self.label_mappings = [[l.encode('utf8') for l in ls]
                               for ls in self.label_mappings]
        self.file_names = [x.encode('utf8') for x in self.file_names]

        with h5py.File(file_name, 'w') as h5_fh:
            activation_table = h5_fh.create_dataset(
                'activations', data=self.activations)
            activation_table.attrs['image_count'] = self.image_count
            activation_table.attrs['neuron_count'] = self.neuron_count
            h5_fh.create_dataset('labels', data=self.labels)
            h5_fh.create_dataset('label_mappings', data=self.label_mappings)
            h5_fh.create_dataset('file_names', data=self.file_names)
            for label in self.activation_labels:
                h5_fh.create_dataset(
                    'activation_labels/{}'.format(label),
                    data=self.activation_labels[label])


class ActivationTableFileSystem(ActivationTable):
    """ variant of activation table that accesses the bottleneck files directly."""

    # pylint: disable=super-init-not-called, too-many-locals
    def __init__(self, target_directory: str) -> None:
        super().__init__()
        """ Constructor. """
        dir_list = next(os.walk(target_directory))[1]
        print(dir_list)
        label_count = len(dir_list)

        image_count = 0
        for root, _, files in os.walk(target_directory):
            image_count += len([x for x in files if x.endswith('.txt')])

        print('image_count:{}'.format(image_count))

        # pick a file, any file
        print('looking for activations')
        arbitrary_file = None
        for filename in os.listdir(
            os.path.join(
                target_directory,
                dir_list[0])):
            print('is it {}?'.format(filename))
            if filename.endswith('.txt'):
                arbitrary_file = filename
                break

        if arbitrary_file is None:
            raise FileNotFoundError("No files found to parse")

        activations = open(os.path.join(
            target_directory, dir_list[0], arbitrary_file)).readline()
        neuron_count = len([float(x) for x in activations.split(',')])

        print(
            'Found {} images across {} categories activating {} neurons'.format(
                image_count,
                label_count,
                neuron_count))

        activation_table = np.empty((image_count, neuron_count), dtype='f')
        label_table = np.array(dir_list)
        label_name_table = np.empty((image_count,), dtype=label_table.dtype)

        label_mapping = {}
        file_names = []

        image_index = 0
        for root, label, files in os.walk(target_directory):
            for activation_file in files:
                if not activation_file.endswith('.txt'):
                    continue
                if activation_file == 'labels.txt':
                    continue
                activation_list = open(os.path.join(
                    root, activation_file)).readline()
                activations = [float(x) for x in activation_list.split(',')]
                activation_table[image_index, :] = activations
                label = root.split('/')[-1]
                # labels.append(root.split('/')[-1])
                label_name_table[image_index] = label
                try:
                    label_mapping[label].append(image_index)
                except KeyError:
                    label_mapping[label] = [image_index]
                file_names.append(activation_file)
                image_index += 1
                if (image_index % 100) == 0:
                    print('read {} images.'.format(image_index))

        self.file_names = np.array(file_names)
        self.activation_labels = {}
        for label in label_mapping:
            self.activation_labels[label] = np.array(label_mapping[label])

        self.activations = activation_table
        self.labels = label_table
        self.label_mappings = label_name_table
        self.neuron_count = neuron_count
        self.image_count = image_count
