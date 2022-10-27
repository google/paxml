# Steps to release a new paxml pip package

Update the version number in setup.py and commit it.

Copy the package using the script `export/copy_to_local_paxml.sh` or through git
clone.

In the following steps, we assume that the repo is stored at /tmp/paxml.

After the docker file has been tested on a TPU VM, the following 
steps use the same Dockerfile and build/run docker on a cloudtop.
Build the docker image for building the pip package

```sh
docker build --tag jax:paxml-pip - < pip_package/Dockerfile
```

Enter the docker image, mapping local directory /tmp/paxml to docker directory /tmp/paxml :

```sh
docker run --rm -it -v /tmp/paxml:/tmp/paxml --name <name> jax:paxml-pip bash

#inside docker
cd /tmp/paxml
```

From the /tmp/paxml directory, run

```sh
rm -rf /tmp/paxml_pip_package_build
PYTHON_MINOR_VERSION=8 pip_package/build.sh
```

If everything goes well, this will produce a wheel for python3.8 in
/tmp/paxml_pip_package_build. You can copy to /tmp/paxml/ and
test scp it to a VM

If this works successfully, you can then upload to the production server.
Remember to update the list of releases in the main README.
