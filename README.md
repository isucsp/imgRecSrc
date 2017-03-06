## Image Reconstruction Source Code

### What does this package contain

#### Beam Hardening Correction Algorithms

All examples and data for our blind beam hardening correction method are
under folder `bhcEx` with filenames ended with `Ex`.  The figures in our
paper can be reproduced by first run `*Ex`, e.g., `yangEx`, followed by
`*Ex('plot')`, e.g., `yangEx('plot')`.  Algorithm implementations are under
`bhc`.

##### References

1. R. Gu and A. Dogandžić, “Blind X-ray CT Image Reconstruction from
   Polychromatic Poisson Measurements,” *IEEE Trans. Comput. Imag.*, vol. 2,
   no. 2, pp. 150–165, 2016.
   [\[DOI\]](http://dx.doi.org/10.1109/TCI.2016.2523431)
   [\[PDF\]](http://isucsp.github.io/imgRecSrc/pdf/beamhardenDouble.pdf)
   [\[Poster\]](http://www.sigport.org/668)

1. R. Gu and A. Dogandžić. (Sep. 2015). Polychromatic X-ray CT Image
   Reconstruction and Mass-Attenuation Spectrum Estimation. *arXiv*:
   [1509.02193](http://arxiv.org/abs/1509.02193).

1. R. Gu and A. Dogandžić, “Beam hardening correction via mass attenuation
   discretization,” in *Proc. IEEE Int. Conf. Acoust., Speech, Signal
   Process.*, Vancouver, Canada, May 2013, pp. 1085–1089.
   [\[DOI\]](http://dx.doi.org/10.1109/ICASSP.2013.6637817)
   [\[PDF\]](http://isucsp.github.io/imgRecSrc/pdf/icassp2013.pdf)
   [\[Poster\]](http://isucsp.github.io/imgRecSrc/pdf/icassp2013poster.pdf)

### How to Install

To install this package, first download the repository by running

    git clone https://github.com/isucsp/imgRecSrc.git

after downloading, from MATLAB change your current folder to `imgRecSrc/`
and execute `setupPath.m` to add necessary paths to the environment.

For X-ray CT examples, the projection and back projection operator
subroutines may be called from MATLAB.  Since they are written in `c`
language, to prepare MATLAB recognizable `MEX` files, go to `imgRecSrc/prj`
and compile the necessary files.  Instructions on compiling the code are
provided for both `UNIX` and `Windows`:

#### For `UNIX`

require: gcc, cuda toolkit (optional) and GPU (optional)

Execute `make cpu` to compile all cpu implementations.  If you have GPU
equipped, run `make gpu` to compile GPU implementation of the X-ray CT
projection operators.  The matlab code will automatically choose to run on
GPU if equipped.

If errors are reported while compiling the `*.c`/`*.cu` files under
`imgRecSrc/prj`, please edit the first few lines in
`imgRecSrc/prj/Makefile` to make sure the path for your `CUDA` installation
is correct.

#### For `Windows`

require: Visual Studio, cuda toolkit (optional) and GPU (optional)

Execute `imgRecSrc/setupPath.m` can automatically compile all needed files
and add paths.

If there is a GPU equipped in your PC, follow the following steps:

* Open the `VS Native Tools Command Prompt` via `Start -> Microsoft Visual
Studio -> Visual Studio Tools`;

* Use `cd` command to change directory to your `imgRecSrc/prj`;

* Run `nvcc -c gpuPrj.cu` to generate the `obj` file;

* Execute `imgRecSrc/setupPath.m`.

### Note

The comments in some of `*.m` files may contain greek letters, which
are `UTF-8` encoded.  Please open in an appropriately configured text
editor.

