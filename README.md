# video-inpainting
This repository contains code for my BSc Thesis done at University of Warsaw. Title of my work was "System for video object removal".

I designed the system for Video Object Removal task, using existing ML models for Object Segmentation and Video Inpainting. System was designed with great flexibility in mind. It was also crucial to allow easy interchangeability of modules, so new/better models or paradigms could be applied.

I found this subject interesting, as it's very demanded, flashy and practical feature which is still not popular or well developed. Work on it was also close to typical ML engineering job, and I enjoy system design in general. It's a bit of shame with as many of well researched tasks not being turned into great products.

## Abstract
The purpose of this thesis was to design a system for video object removal. Objects are defined as people, animals, items, or other often existing elements that make up the foreground or background of the recording. A complex, configurable framework was proposed, using state-of-the-art solutions in the machine learning field, more specifically deep learning. The designed system was divided into two parts: object selection and video inpainting. Integration of these parts has been implemented. While designing the system, it was crucial to consider many use cases, which were later ensured in the system. It was noticed, that video object removal is a broad topic and it is dependent on a specific recording. Three main types of inpainted recordings were classified: static recordings, dynamic recordings, and recordings with stationary objects. Methods used in this system allow the user to obtain desired results for the first two defined video types, with room for improvement for the most challenging stationary objects. A solution to this problem has been proposed, but it is not implemented in the current system state. In addition to existing methods for video inpainting, a new method has been implemented, which extends one of the previous methods. The introduced method explicitly allows to obtain intact source recordings excluding removed parts. It is essential in a use case, where the system is used for removal of unwanted objects while keeping the original appearance of foreground elements. Because of the limitations of specific methods, only providing multiple solutions allowed for system configuration for specific recordings, thus creating a complex and universal framework. The proposed system allows interchangeability of modules for specific subtasks with minimal integration cost. This approach allows applying significant modifications without changing of overall system structure, which is a key factor, because of

## Future work
I guess the main biggest improvemenet would be the addition of generated images (with Image Inpanting model).

Currently, Video Ipainting models only focus on "spreading" information about filled regions from other video frames (for eg. using optical flow), without "generation" of type typical for image inpainting models. Image inpainting models generate high-quality and realistic patches, but obviously applying them separetly to each image would result in non-cocherent video. Only spreading information helps achieve high level of cocherency between frames, making results look natural to human eye.

Hovever, I think that addition of generated frames would be highly beneficial in cases, where there is no relevant information to spread to inpainted area and model starts hallucinating low-quality patches. We could generate additional images with Image Inpainting before proper Video Inpainting.

Imagine video, where we have statical object, permamently shadowing some region of the scene. Using my work, removal of this object would give low-quality background. However, imagine generating one additional frame with generated patch for this occupied region. Now Video Inpainting model could relevantly fill corresponding regions with high-quality content.

There should be some metric method to determine if there is already enough information about region we want to inpaint in video. If not, we should add some inpainted images :smile:.

I wish I had more time before the deadlines to check it out on my own, as I find it interesting and somehow novel (it's not well described anywhere). If you use Adobe AE Content Aware Fill or Runway Image Inpainting, you can notice similar approach applied with high success.

## Thesis access
Thesis alone was published in Polish (only Abstract is English).
You can access it [here](https://drive.google.com/file/d/1aAlqlGB46G4onAiFOnWsknW16P30CtCD/view?usp=sharing) or read more about it in [Base of Knowledge of Warsaw University of Technology](https://repo.pw.edu.pl/info/bachelor/WUT55d2d5c709a94d35b2e8a0aa792b0ba9/).
