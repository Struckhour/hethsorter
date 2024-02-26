# HethSorter
## --_a program for sorting and labelling Hermit Thrush Song_--

### Overview

HethSorter takes wave files along with Raven selection tables (in which hermit thrush songs have been selected but not labelled) and returns selection tables with labelled songs. For better labeling accuracy, HethSorter can also reference previously labelled files for a given bird.
	
**Inputs:**
 - .wav file with hermit thrush song
 - a corresponding .txt [Raven] selection table with accurately selected hermit thrush intro notes
 - the hethsorter_params.csv file for designated the filenames. For better accuracy, include a wave file and selection table that has labelled song-types.

**Output:**
 - a .txt [Raven] selection table with songs categorized and labelled by song-type.

### Installation

Currently, HethSorter requires that python be installed on the user’s computer. HethSorter was developed on python version 3.9.12 but other versions are likely to be compatible. Other python libraries may need to be installed as well, including tensorflow and librosa. I recommend using anaconda to manage python and its dependencies. To download anaconda and learn more, go to: https://www.anaconda.com/download

### Usage

There are two contexts for using HethSorter, each with its own workflow. The two contexts are:
1)	**Without Templates**: Using HethSorter without ever having labelled the focal bird’s song-types.
2)	**With Templates**: Using HethSorter when the focal bird’s song-types have been previously labelled on another recording.

##### Without Templates Workflow
**_Setup_**
1.	Begin with a folder that contains the following files:
    * the wave file to be analyzed.
    * a corresponding Raven selection table with accurate intro-note timings
    * the HethSorter.py file.
    * the HethSorter_params.csv file
2.	Open the HethSorter_params.csv file and enter the .wav filename(s) into the filename column. Enter all filenames to be analyzed. Do not change anything else about this file, such as the format or columns.
3.	Enter the corresponding selection table filename in the selection table column next to the corresponding wave filename.

**_Run HethSorter_**
1. Open HethSorter.py in whatever code editor you use. Popular options are VSCode, Idle, or Jupyter Labs.
2. Run HethSorter. It should take less than a minute for a wave file that is less than an hour long. HethSorter will produce a new .txt file for each .wav file it analyzes. These can be opened in Raven as selection tables.

**_Review and Manually Edit_**
1. Open each new selection table and wave file in Raven. Check the selections for labelling errors and fix them if need be. I recommend ordering selections by ID and/or frequency and scrolling through them so that similar songs are grouped together. It is unlikely that HethSorter will have correctly labelled all the songs. This stage is meant as a time-saver to get the labelling started. Even trained researchers sometimes struggle to categorize a given song-type.

##### With Templates Workflow
**_Setup_**
1.	Begin with a folder that contains the following files:
    * the HethSorter.py file.
    * the HethSorter_params.csv file
    * the wave file to be analyzed. All recordings need to be from the same bird and the column for songtype ID must be named: ID
    * Raven selection table with accurate intro-note timings (but no labelled song-types)
    * A wave file that has a corresponding Raven selection table with accurately labelled song-types
    * The corresponding Raven selection table with accurate intro-note timings and labelled song-types.
2.	Open the HethSorter_params.csv file and enter the .wav filename into the filename column. Do not change add or remove columns or change column names.
3.	Enter the corresponding to-be-labelled selection table filename in the selection table column next to the corresponding wave filename. Do not include the extension.
4.	Enter the template wave file name in the third column (template_wave).
5.	Enter the template selection table name in the fourth column (template_table). 

**_Run HethSorter_**
1.	Open HethSorter.py in whatever code editor you use. Popular options are VSCode, Idle, or Jupyter Labs.
2.	Run HethSorter. It should take less than a minute for a wave file that is less than an hour long. HethSorter will produce a new .txt file for each .wav file it analyzes. These can be opened in Raven as selection tables.

**_Review and Manually Edit_**
1.	Open the new selection table and wave file in Raven. Check the selections for labelling errors and fix them if need be. I recommend ordering selections by ID and/or frequency and scrolling through them so that similar songs are grouped together. With a template file, HethSorter should be very accurate (above 95%) unless it is a very noisy recording or there are song-types that were not included in the template file. Nonetheless, it is recommended that a researcher checks the file, especially if high accuracy is required for the research design.

### How it Works
**Without Templates**: HethSorter takes a picture of the spectrogram at every timestamp and first filters out as much noise as possible in every picture. It then reduces the resolution of the picture. This is because song-types often differ from each other in minor ways, such as slight shifts in pitch or timing. Reducing the resolution blurs away these finer differences without losing the general shape of the song-type.

It then iterates through each picture, measuring the differences between them, and grouping the most similar songs together. It continues to collapse the most similar groups with each other until it reaches a difference threshold. This is called hierarchical clustering and is used because the number of groups is unknown. The difference threshold has been empirically selected to err on the side of too many song-type categories rather than too few. This is because it is easier to manually collapse groups than split groups.

Programmatically selecting the correct number of groups is a popular problem for clustering algorithms and has multiple imperfect approaches. I experimented with some different K-means clustering approaches, but for simplicity’s sake, and because every bird has its own repertoire of song-types, HethSorter uses an arbitrary cutoff. This is likely an area where the program could be improved. Machine Learning might be possible but has its own set of challenges because every individual bird’s repertoire has unique song-types and a different number of song-types.

**With Templates**: HethSorter first goes to the template wave and template selection table and creates a filtered, low-resolution, average image for every song-type category. It then goes through the unlabelled spectrogram at each timestamp and compares the filtered, low-resolution image at each song to each of the template images. A song takes the label of the template that it best matches.

### Contact and Additional Info:

**Contact**: HethSorter was created by Luke McLean (MA candidate) for the Sean Roach Laboratory at the University of New Brunswick – Saint John Campus. Inquiries about the software can be sent to lmclean@unb.ca 

**Future Versions**: It is my hope that I will continue to improve HethSorter as I use it and find time to work on it. The code is publicly available on [github]. If you’d like to help or have questions, don’t hesitate to contact me.

**HethFinder**:  Sorting songs is merely the second and easier part of analyzing hermit thrush song. Before HethSorter sorts the songs, they need to be found and selected. So, if you have raw, unlabeled wave files of recorded hermit thrush, be sure to check out HethSorter’s best friend and companion, [HethFinder].

[Raven]: https://www.ravensoundsoftware.com/software/raven-pro/
[github]: https://github.com/Struckhour/hethsorter
[HethFinder]: https://github.com/Struckhour/hethfinder
