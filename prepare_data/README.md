# Download and Preprocess KITTI data

This script downloads, and prepares the KITTI data to be used to train and evaluate the SfM-Net.

The data is downloaded, extracted, and bundled into TFRecords. Note that due to the large amount of data, this may take up to 400GB of disk space, and upwards of 24 hours to complete.

## Preparation

Before starting the script, make sure you have enough disk space and that your computer will be able to run uninterrupted for a long period of time.

Additionally, download the required packages using `pip3 install -r requirements.txt`.

## Running the script

Run the script with the command `python3 prepare_data.py <save_path>`.

Note the final dataset is on the order of 2GB, so it may be possible for me to transfer it to you without having to run the script.

## References

Any work using the KITTI dataset should cite the original authors.

	@ARTICLE{Geiger2013IJRR,
		author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
        title = {Vision meets Robotics: The KITTI Dataset},
        journal = {International Journal of Robotics Research (IJRR)},
        year = {2013}
    }
