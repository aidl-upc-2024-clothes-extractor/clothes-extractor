import os
import zipfile
import requests

class DatasetUtil():
    _download_url = "https://www.dropbox.com/s/10bfat0kg4si1bu/zalando-hd-resized.zip?dl=0"
    _dataset_path = "datasets"
    _output_path = os.path.join(_dataset_path, "zalando-hd-resized")
    _download_tmp_file = os.path.join(_dataset_path, "zalando-hd-resized.zip")

    def __init__(self, download_url=None, output_path=None):
        self._download_url = download_url if download_url is not None else self._download_url
        self._output_path = output_path if output_path is not None else self._output_path

    def start(self):
        if os.path.exists(self._output_path):
            return True
        # self.download()
        self.unzip_dataset()

    def download(self):
        response = requests.get(self._download_url, stream=True)
        if response.status_code == 200:
            with open(self._download_tmp_file, 'wb') as file:
                for chunk in response.iter_content(chunk_size=128):
                    file.write(chunk)
            print(f"Downloaded successfully to {self.destination}")
        else:
            raise Exception(f"Failed to download. Status code: {response.status_code}")


    def prepare_images(self):
        pass

    def unzip_dataset(self):
        if not os.path.exists(self._output_path):
            # os.mkdir(extraction_path)
            print("Extracting dataset...")
            with zipfile.ZipFile(self._download_tmp_file, 'r') as zip_ref:
                zip_ref.extractall(self._dataset_path)
            print("...Done")