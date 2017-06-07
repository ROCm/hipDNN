# hipDNN

This is work in progress. Current status:
1. The makefiles are still in amateur state. You can expect to edit them to your particular configuration if you want to compile this code. This needs to be improved significantly (help is welcome!)

2. There is no hipify, although hipification is trivial enough.  Main steps:
+ Search and replace cudnn with hipdnn (typically for function calls and descriptors).
+ Search and replace CUDNN with HIPDNN (typically for enumerated types).
+ Include hipDNN.h, and link the DSO hipDNN.so  (currently you need to compile the scr file for the platform, no DSO yet).

2. There are known issues in MIOpen that need to be addressed, not only for hipDNN, but in general. This problematic code is currently disabled in hipDNN as it would not compile. You can find such code by searching the string NOTYET. Generally speaking, issues fall into the following categories: 

+ Some miopen API calls define descriptors as "const" while they shouldn't be, or do not define them as const when const is appropriate.

+ Some miopen API calls do not treat workspace well, for example they are missing workspaceSize.

+ miopen provides the "Ex" version of some cudnn calls. This would be fine, except miopen does not export a way to discover the device pointer from the descriptor (to be added).

3. There are known thigns but need to be improved besides MIOpen. Some of those are noted as HGSOS, mostly "notes to self" that should disappear over time.
