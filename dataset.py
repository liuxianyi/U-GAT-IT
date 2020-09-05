import os
import sys
import cv2
from PIL import Image
import numpy as np
# from paddle.io import Dataset

__all__ = ["DatasetFolder", "ImageFolder"]

class Dataset(object):
    """
    An abstract class to encapsulates methods and behaviors of datasets.
    All datasets in map-style(dataset samples can be get by a given key)
    should be a subclass of `paddle.io.Dataset`. All subclasses should
    implement following methods:
    :code:`__getitem__`: get sample from dataset with a given index. This
    method is required by reading dataset sample in :code:`paddle.io.DataLoader`.
    :code:`__len__`: return dataset sample number. This method is required
    by some implements of :code:`paddle.io.BatchSampler`
    see :code:`paddle.io.DataLoader`.
    Examples:
        
        .. code-block:: python
            import numpy as np
            from paddle.io import Dataset
            
            # define a random dataset
            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples
            
                def __getitem__(self, idx):
                    image = np.random.random([784]).astype('float32')
                    label = np.random.randint(0, 9, (1, )).astype('int64')
                    return image, label
                
                def __len__(self):
                    return self.num_samples
            
            dataset = RandomDataset(10)
            for i in range(len(dataset)):
                print(dataset[i])
    """

    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__len__', self.__class__.__name__))

def has_valid_extension(filename, extensions):
    """Checks if a file is a vilid extension.
    Args:
        filename (str): path to a file
        extensions (tuple of str): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(dir, class_to_idx, extensions, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)

    if extensions is not None:

        def is_valid_file(x):
            return has_valid_extension(x, extensions)

    #for target in sorted(class_to_idx.keys()):
    #d = os.path.join(dir, target)
    # if not os.path.isdir(d):
    #     continue
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                item = (path, 0)
                images.append(item)

    return images
    #     images = []
#     for root, _, fnames in sorted(os.walk(dir)):
#         for fname in sorted(fnames): # 获取文件夹下的所有图片文件，自动过滤非图片文件
#             if has_file_allowed_extension(fname, extensions):
#                 path = os.path.join(root, fname)
#                 item = (path, 0)
#                 images.append(item)

#     return images


class DatasetFolder(Dataset):
    """A generic data loader where the samples are arranged in this way:
        root/class_a/1.ext
        root/class_a/2.ext
        root/class_a/3.ext
        root/class_b/123.ext
        root/class_b/456.ext
        root/class_b/789.ext
    Args:
        root (string): Root directory path.
        loader (callable|optional): A function to load a sample given its path.
        extensions (tuple[str]|optional): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable|optional): A function/transform that takes in
            a sample and returns a transformed version.
        is_valid_file (callable|optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self,
                 root,
                 loader=None,
                 extensions=None,
                 transform=None,
                 is_valid_file=None):
        self.root = root
        self.transform = transform
        if extensions is None:
            extensions = IMG_EXTENSIONS
        #classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, None, extensions,
                               is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError(
                "Found 0 files in subfolders of: " + self.root + "\n"
                "Supported extensions are: " + ",".join(extensions)))

        self.loader = cv2_loader if loader is None else loader
        self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = samples
        #self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), 
                    and class_to_idx is a dictionary.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [
                d for d in os.listdir(dir)
                if os.path.isdir(os.path.join(dir, d))
            ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        mean = np.array([0.5,0.5,0.5], dtype=np.float32).reshape(1, 1,3)
        std = np.array([0.5,0.5,0.5], dtype=np.float32).reshape(1, 1,3)
        sample = (sample/255.0-mean)/std
        return sample, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def cv2_loader(path):

    return cv2.imread(path)
    # with open(path, 'rb') as f:
    #     img = Image.open(f)
    #     img = img.convert('RGB')
    #     return np.array(img).transpose((1,2,0))


class ImageFolder(Dataset):
    """A generic data loader where the samples are arranged in this way:
        root/1.ext
        root/2.ext
        root/sub_dir/3.ext
    Args:
        root (string): Root directory path.
        loader (callable, optional): A function to load a sample given its path.
        extensions (tuple[string], optional): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        samples (list): List of sample path
     """

    def __init__(self,
                 root,
                 loader=None,
                 extensions=None,
                 transform=None,
                 is_valid_file=None):
        self.root = root
        if extensions is None:
            extensions = IMG_EXTENSIONS

        samples = []
        path = os.path.expanduser(root)

        if extensions is not None:

            def is_valid_file(x):
                return has_valid_extension(x, extensions)

        for root, _, fnames in sorted(os.walk(path, followlinks=True)):
            for fname in sorted(fnames):
                f = os.path.join(root, fname)
                if is_valid_file(f):
                    samples.append(f)

        if len(samples) == 0:
            raise (RuntimeError(
                "Found 0 files in subfolders of: " + self.root + "\n"
                "Supported extensions are: " + ",".join(extensions)))

        self.loader = cv2_loader if loader is None else loader
        self.extensions = extensions
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path)
 
        if self.transform is not None:
            sample = self.transform(sample)
        return [sample]

    def __len__(self):
        return len(self.samples)






# # import torch.utils.data as data
# from PIL import Image
# import os
# import os.path



# def has_file_allowed_extension(filename, extensions):
#     """Checks if a file is an allowed extension.
#     检测图片格式
#     Args:
#         filename (string): path to a file

#     Returns:
#         bool: True if the filename ends with a known image extension
#     """
#     filename_lower = filename.lower()
#     return any(filename_lower.endswith(ext) for ext in extensions)





# def find_classes(dir):
#     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#     classes.sort()
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return classes, class_to_idx


# def make_dataset(dir, extensions):
#     images = []
#     for root, _, fnames in sorted(os.walk(dir)):
#         for fname in sorted(fnames): # 获取文件夹下的所有图片文件，自动过滤非图片文件
#             if has_file_allowed_extension(fname, extensions):
#                 path = os.path.join(root, fname)
#                 item = (path, 0)
#                 images.append(item)

#     return images


# def get_random_images_and_labels(data,transform=None, target_transform=None):
#     """
#     加载一张图片
#     """
#     path, target = data
#     sample = default_loader(path)
#     if transform is not None:
#         sample = transform(sample)
#     if target_transform is not None:
#         target = target_transform(target)

#     return sample, target

# # If the data generator yield list of samples each time,
# # use DataLoader.set_sample_list_generator to set the data source.
# def sample_list_generator_creator(BATCH_SIZE, dir,transform=None, target_transform=None):
#     """
#     加载a batch
#     """
#     images = make_dataset(dir, IMG_EXTENSIONS)
#     def __reader__():
#         for image in images:
#             sample_list = []
#             for _ in range(BATCH_SIZE):
#                 image, label = get_random_images_and_labels(image,transform, target_transform)
#                 sample_list.append([image, label])

#             yield sample_list

#     return __reader__
# # class DatasetFolder(data.Dataset):
# #     def __init__(self, root, loader, extensions, transform=None, target_transform=None):
# #         # classes, class_to_idx = find_classes(root)
# #         samples = make_dataset(root, extensions) # (N, path, 0)
# #         if len(samples) == 0:
# #             raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
# #                                "Supported extensions are: " + ",".join(extensions)))

# #         self.root = root # 文件夹路径
# #         self.loader = loader # 解析路径成图片数据 PIL
# #         self.extensions = extensions # 满足格式的后缀
# #         self.samples = samples # (N, path, 0)

# #         self.transform = transform # 图片数据增强
# #         self.target_transform = target_transform # 图片数据增强

# #     def __getitem__(self, index):
# #         """
# #         Args:
# #             index (int): Index

# #         Returns:
# #             tuple: (sample, target) where target is class_index of the target class.
# #         """
# #         path, target = self.samples[index]
# #         sample = self.loader(path)
# #         if self.transform is not None:
# #             sample = self.transform(sample)
# #         if self.target_transform is not None:
# #             target = self.target_transform(target)

# #         return sample, target

# #     def __len__(self):
# #         return len(self.samples)

# #     def __repr__(self):
# #         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
# #         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
# #         fmt_str += '    Root Location: {}\n'.format(self.root)
# #         tmp = '    Transforms (if any): '
# #         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
# #         tmp = '    Target Transforms (if any): '
# #         fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
# #         return fmt_str


# IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


# def pil_loader(path):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')


# def default_loader(path):
#     return pil_loader(path)

# def set_data_source(loader, places,BATCH_SIZE, dir,transform=None, target_transform=None):
#     loader.set_sample_list_generator(sample_list_generator_creator(BATCH_SIZE, dir,transform, target_transform), places=places)

# # class ImageFolder(DatasetFolder):
# #     def __init__(self, root, transform=None, target_transform=None,
# #                  loader=default_loader):
# #         super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
# #                                           transform=transform,
# #                                           target_transform=target_transform)
# #         self.imgs = self.samples
