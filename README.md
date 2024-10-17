# rawplot
 
 Collection of plotting commands to analyze RGB imagers producing Bayer RAW images, such as DSLR cameras or other imagers (i.e RasPi HQ camera) using matplotlib. Its purpose is helping characterize the camera sensor performance and key figures such as gain and read noise.

 This is a child project of [AZOTEA](https://guaix.ucm.es/azoteaproject), an initiative to monitor light pollution through digital imagers. The AZOTEA initiative was started in EU funded project ACTION - Participatory science toolkit against pollution, Grant 824603.


 ## Installation

It is haighly recommended to create a virtual environment and activate it, then install rawplot from PyPi

```bash
$ mkdir rawpplot
$ cd rawplot
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install rawplot

```
# Usage

The available commands list can be found in the `bin` virtual environment sub-folder.

 ```bash
 ls -1 .venv/bin/raw*
 venv/bin/rawplot-filters
.venv/bin/rawplot-hv
.venv/bin/rawplot-image
.venv/bin/rawplot-imarith
.venv/bin/rawplot-linearity
.venv/bin/rawplot-master
.venv/bin/rawplot-photodiode
.venv/bin/rawplot-plan
.venv/bin/rawplot-ptc
.venv/bin/rawplot-snr
.venv/bin/rawplot-spectral
 ```

## Common options

All utilities share these command line options, mainly used for debugging purposes:

* `⁻h, --help` shows program's options and exit
* `--version` show program's version numbe and exit.
* `--console` log debugging messages to console. Default level is `INFO`.
* `--log-file` log debugging messages to file.  Default level is `INFO`.
* `--verbose`  raises debugging level to `DEBUG`.
* `--quiet` lowers debugging level to `WARN`.

## Other common options using in many commands

### subsampling

* `--every` <N> Picks or subsamples an object retaled with the command (pick every N images in a file list, subsamples points in a histogram by /N, etc.)

### Color planes selection

Since we are dealing with RAW images, we are dealing with 4 separate color planes (channels): `R Gr Gb B`. Most of the commands supports being run on all or a subset of these color planes.

* `-c | --channel` Specify which of the R, Gr, Gb, or B channels to handle by the command. Can be one or a combination of them. Some commands accept a G channel
(an average of Gr and Gb)

Order in the command line is not important. They are internally reordered so that the display & processing order is (R, Gr, Gb, B)

Valid Examples:
```bash
--channel B
--channel Gr Gb   
--channel R G B
--channel Gr R B Gb
```
Invalid Examples:

```bash
--channel B B (duplicated)
--channel R A (A is not supported)
--channel R G B Gb (this commbination is not supported)
```

### Region of Interest (ROI)

A ROI is specified given its starting point `(x0, y0)` and dimensions `(width, height)`. To unifomingly address the different resolutions in camera formats, a normalized ROI is used as input, where both the starting point and dimensions are normalized between 0 and 1. In addition `x0+width <= 1.0` and `y0+height <= 1.0`. When the `(x0,y0)` starting point is not specified in the command line, the `(width,height)` dimensions are assumed with respect to the image center.

The different ROI parameters on the command line can be specified as either as decimals or fractions for convenience.
Width and height parameters can be specified with `-wi | --width` and `-he | --height`.

Example:
```
--x0 0.2 --y0 0.1 --width 1/3 --height 1/4 
```

Only when an image file is open, this normalized ROI transforms into a physical ROI with pixels. When the physcal ROI is displayed in several graphs, it is done using [NumPy](https://numpy.org/) matrix style `[y0:y1,x0:x1]`.

# Commands

Examples of usage can be found in chapter 7 (Tools) of [GONet all sky camera calibration](https://doi.org/10.5281/zenodo.11183813) report along with the [dataset on which the usage examples are based](https://doi.org/10.5281/zenodo.11125041).

The camera itself was a [Raspberry Pi HQ Camera](https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera/), installed in a [GoNET project](https://www.researchgate.net/publication/351459667_The_GONet_Ground_Observing_Network_Camera_An_Inexpensive_Light_Pollution_Monitoring_System) device.


![Raspberry Pi HQ camera flat image](doc/images/flat_image.png)

![Raspberry Pi HQ camera flat image histogram](doc/images/flat_histo.png)

![Raspberry Pi HQ Camera linearity plot](doc/images/linearity.png)

![Raspberry Pi HQ HV Spectrogram](doc/images/hv.png)

## Photon Transfer Curves (PTC)

An series of PTC charts, based on the classic [Photon Transfer](https://www.spiedigitallibrary.org/ebooks/PM/Photon-Transfer/eISBN-9780819478382/10.1117/3.725073#_=_) book have been included so far:

|  CURVE   |                Description               | Units                       |
| :------: | :--------------------------------------- | :-------------------------- |
| Curve 1  | read, shot, FPN & total noise vs. signal | log rms DN vs. log DN       |
| Curve 1  | read, shot, FPN & total noise vs. signal | log rms $e^-$ vs. log $e^-$ |
| Curve 2  | read + shot noise vs. signal             | log rms DN vs. log DN       |
| Curve 2  | read + shot noise vs. signal             | log rms $e^-$ vs. log $e^-$ |
| Curve 3  | shot noise vs. signal                    | log rms DN vs. log DN       |
| Curve 3  | shot noise vs. signal                    | log rms $e^-$ vs. log $e^-$ |
| Curve 4  | FPN vs. signal                           | log rms DN vs. log DN       |
| Curve 4  | FPN vs. signal                           | log rms $e^-$ vs. log $e^-$ |
| Curve 5  | Read + Shot Noise Variance vs. signal    | DN vs. DN                   |
| Curve 6  | SNR vs. signal                           | log SNR vs. log DN          |
| Curve 6  | SNR vs. signal                           | log SNR vs. log $e^-$       |           


The curves are based on the following simplified detector model, whose noise variances given by:

```math
\sigma_{TOTAL}^2 = \sigma_{READ}^2 + \sigma_{SHOT}^2 + \sigma_{FPN}^2 \quad [e^-]
```

Where $\sigma_{READ}$ is the detector read noise, $\sigma_{SHOT}$ is the signal-dependent Poisson noise ($\sigma_{SHOT} = S^{1/2}$), and $\sigma_{FPN}$ is a fixed spatial pattern noise, not all pixels being equal, being also modeled as a signal-dependent noise $\sigma_{FPN} = P_{FPN}S$, where $P_{FPN} \ll 1$.

Working with digital numbers [DN] - instead of electrons - equation above is rewritten as:

```math
\sigma_{TOTAL}^2 = \sigma_{READ}^2 + (S/g) + (P_{FPN}S)^2 \quad [DN]
```

## Miscellaneous Commands

Convenicence commands to help processing an analysis of images using the above commands.

### rawplot-master

Utility to make master bias, dark or flat frames from a series of RAW files. Produces a 3D FITS cube, one layer per color.


```bash
rawplot-master --console --input-dir images/20240124/biases/ --filter bias* --batch 5 --prefix master --image-type bias
```

Produces the following result:


```bash
2024-02-03 12:38:17,855 [INFO] ============== rawplot.master 0.1.dev89+gc34abee ==============
2024-02-03 12:38:17,855 [INFO] Normalized ROI is [P0=(0.0000,0.0000) DIM=(1.0000 x 1.0000)]
2024-02-03 12:38:17,921 [INFO] The process comprises 8 batches of 5 images max. per batch
2024-02-03 12:38:17,921 [INFO] [1/8] Begin loading 5 images into RAM with R Gr Gb B channels, 2028 x 1520 each
2024-02-03 12:38:18,343 [INFO] [2/8] Begin loading 5 images into RAM with R Gr Gb B channels, 2028 x 1520 each
2024-02-03 12:38:18,727 [INFO] [3/8] Begin loading 5 images into RAM with R Gr Gb B channels, 2028 x 1520 each
2024-02-03 12:38:19,124 [INFO] [4/8] Begin loading 5 images into RAM with R Gr Gb B channels, 2028 x 1520 each
2024-02-03 12:38:19,513 [INFO] [5/8] Begin loading 5 images into RAM with R Gr Gb B channels, 2028 x 1520 each
2024-02-03 12:38:19,892 [INFO] [6/8] Begin loading 5 images into RAM with R Gr Gb B channels, 2028 x 1520 each
2024-02-03 12:38:20,274 [INFO] [7/8] Begin loading 5 images into RAM with R Gr Gb B channels, 2028 x 1520 each
2024-02-03 12:38:20,658 [INFO] [8/8] Begin loading 1 images into RAM with R Gr Gb B channels, 2028 x 1520 each
2024-02-03 12:38:20,794 [INFO] Saving master bias file from 36 images in /home/rafa/repos/own/lica/rawplot/master_bias_frame_aver.fit
```

### rawplot-imarith

Basic image manipulation command. Can be expanded conveniently

```bash
rawplot-imarith -h

Arithmetic operations on one or two 3D-FITS cubes

positional arguments:
  {sub}
    sub              Substracts an image or a value (second argument) from a given image (first argument)

options:
  -h, --help         show this help message and exit
  --version          show program's version number and exit
  --console          Log to console.
  --log-file <FILE>  Log to file.
  --verbose          Verbose output.
  --quiet            Quiet output.
  --modules MODULES  comma separated list of modules to activate debug level upon.

```

```bash
rawplot-imarith --console sub master_bias_frame_aver.fit 256

2024-02-07 14:55:43,821 [INFO] ============== rawplot.imarith 0.9.1.dev4+gf89ffa4.d20240207 ==============
2024-02-07 14:55:43,891 [INFO] Created result image on: master_bias_frame_aver_subs.fit
```

# Some example charts produced

![Raspberry Pi HQ Camera flat image](doc/image/flat_image.png)

![Raspberry Pi HQ Camera flat image histogram](doc/image/flat_histo.png)

![Raspberry Pi HQ Camera linearity plot](doc/image/linearity.png)

![Raspberry Pi HQ Camera HV Spectrogram](doc/image/hv.png)

![Raspberry Pi HQ PTC Chart](doc/image/ptc.png)
