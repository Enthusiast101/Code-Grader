
import io
import tarfile
import docker


class DockerSandbox:
    def __init__(self):
        self.client = docker.from_env()
        self.container = None

    def start(self):
        self.container = self.client.containers.run(
            "agent-sandbox:latest",
            name="agent-sandbox",
            detach=True,
            command="sleep infinity",
            working_dir="/workspace",
            network_mode="none",
            mem_limit="512m",
            nano_cpus=1_000_000_000,
            pids_limit=64,
            user="10001:10001",
            read_only=True,
            tmpfs={"/workspace": "rw,size=256m,uid=10001,gid=10001"},
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
        )

    def run(self, command):
        if not self.container:
            raise RuntimeError("Start the sandbox first.")

        result = self.container.exec_run(
            command,
            workdir="/workspace",
            user="10001:10001",
        )

        return {
            "exit_code": result.exit_code,
            "output": result.output.decode("utf-8", errors="replace"),
        }

    def write_file(self, filename, content):
        data = content.encode("utf-8")
        archive = io.BytesIO()

        with tarfile.open(fileobj=archive, mode="w") as tar:
            file = tarfile.TarInfo(filename)
            file.size = len(data)
            file.mode = 0o600
            tar.addfile(file, io.BytesIO(data))

        archive.seek(0)

        if not self.container.put_archive("/workspace", archive):
            raise RuntimeError("Could not write file.")

    def read_file(self, filename):
        files, _ = self.container.get_archive(f"/workspace/{filename}")
        archive = io.BytesIO(b"".join(files))

        with tarfile.open(fileobj=archive) as tar:
            file = tar.extractfile(tar.getmembers()[0])
            return file.read().decode("utf-8")

    def close(self):
        if self.container:
            self.container.remove(force=True)
            self.container = None

        self.client.close()
