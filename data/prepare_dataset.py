class DatasetPreparator:
    def __init__(self, dataset_path, train_dir='train', val_dir='val', test_dir='test'):
        self.dataset_path = dataset_path
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.prepare_directories()  

    def prepare_directories(self):
        import os
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def split_dataset(self):
        import os
        import shutil
        import random
        from tqdm import tqdm

        # Get all image file paths
        all_images = [f for f in os.listdir(self.dataset_path) if f.endswith(('.jpg', '.png'))]
        random.shuffle(all_images)

        # Split data
        total = len(all_images)
        train_count = int(total * 0.8)
        val_count = int(total * 0.1)

        train_images = all_images[:train_count]
        val_images = all_images[train_count:train_count + val_count]
        test_images = all_images[train_count + val_count:]

        # Copy images to respective directories
        self.copy_images(train_images, self.train_dir)
        self.copy_images(val_images, self.val_dir)
        self.copy_images(test_images, self.test_dir)

    def copy_images(self, image_list, destination):
        for image in tqdm(image_list, desc=f'Copying to {destination}'): 
            shutil.copy(os.path.join(self.dataset_path, image), destination)

    def verify_preparation(self):
        import os
        assert len(os.listdir(self.train_dir)) == int(len(all_images) * 0.8), "Train set not prepared correctly"
        assert len(os.listdir(self.val_dir)) == int(len(all_images) * 0.1), "Validation set not prepared correctly"
        assert len(os.listdir(self.test_dir)) == len(all_images) - len(os.listdir(self.val_dir)) - len(os.listdir(self.train_dir)), "Test set not prepared correctly"


# Example usage:
# preparator = DatasetPreparator(dataset_path='path_to_celebA')
# preparator.split_dataset()
# preparator.verify_preparation()