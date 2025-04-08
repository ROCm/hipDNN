# hipDNN-sdk
## Schema based data objects
- The hipDNN sdk is using schema based https://flatbuffers.dev/ data objects for describing the graph, and operations.

## How to change schema files
- Adding, or updating the files inside schemas (*.fbs) requires regenerating the output files.
- To regerneate the output files run the make target `make generate_hipdnn_sdk_headers`.
- This will regenarate the files inside include.
- Note: Any changes made to *.fbs files will require regenerating the files, please run the make command anytime the schema files are updated.