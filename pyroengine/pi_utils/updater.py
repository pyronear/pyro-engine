import subprocess


class Updater:
    """Class to update the software on the raspberries."""

    def __init__(self, pyroapi_url, pyroengine_version_path, ansible_directory):
        """
        Params:
        - pyroapi_url: str
            Url of the pyronear api
        - pyroengine_version_path: str
            Path to the text file which contains the hash of the current version of pyro-engine
        - ansible_directory: str
            Path to the directory containing ansible related files on the main rpi (/home/pi/ansible_main)
        """
        self.pyroapi_url = pyroapi_url
        self.pyroengine_version_path = pyroengine_version_path
        self.ansible_directory = ansible_directory
        self.update = False

    def get_new_hash(self):
        """
        Ask the pyronear api for the hash the device should have.

        Return:
            str: the new hash, provided by the pyronear api.
        """
        raise NotImplementedError

    def compute_difference(self, new_hash):
        """
        Compute the difference between the new_hash given by the pyronear api and the
        current version of the hash stored in self.file_path
        """
        with open(self.pyroengine_version_path) as file:
            current_hash = file.read()
        if new_hash != current_hash:
            self.update = True

    def update_command(self, playbook_name):
        """
        Update command to be run.
        """
        return subprocess.run(
            [
                f"{self.ansible_directory}/venv/bin/ansible-playbook",
                f"{self.ansible_directory}/playbooks/{playbook_name}",
            ],
            cwd=self.ansible_directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    def launch_update(self, playbook_name, logs_file_name):
        """
        Launch update and try 3 times maximum. If update fails i.e. return code of the command is 1,
        the logs are stored in a text file under the name `logs_file_name` and returns False.
        If return code is 0, returns True.
        """
        for _ in range(3):
            outpout_rpi = self.update_command(playbook_name)
            if outpout_rpi.returncode != 0:
                logs = outpout_rpi.stdout
                with open(f"{self.ansible_directory}/{logs_file_name}", "a") as file:
                    file.write(logs)
            else:
                return True
        return False

    def run(self):
        """
        Core method of the updater.

        Retrieve the new hash from the pyronear api.
        Compute the difference between the latter and the one stored in the file at `pyroengine_version_path`.
        If the hashes are different, the update is launched.
        First, the update of the main rpi is run. Upon success, the update of the camera rpi is launched.
        If the update of the main rpi has failed, nothing is launched. One will be to debug by hand.
        """
        new_hash = self.get_new_hash()
        self.compute_difference(new_hash)
        if self.update:
            output_main = self.launch_update(
                "update_rpi_main.yml", "logs_update_myself.txt"
            )
            if output_main:
                self.launch_update(
                    "update_rpi_camera.yml", "logs_update_rpi_camera.txt"
                )


if __name__ == "__main__":
    pyroapi_url = "pyroapi"
    pyroengine_version_path = "/home/pi/pyroengine_version.txt"
    ansible_directory = "/home/pi/ansible_main"
    updater = Updater(pyroapi_url, pyroengine_version_path, ansible_directory)
    updater.run()
