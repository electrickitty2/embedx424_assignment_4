
#
# tflite models
#
MODELS		=				\
	custom_objects_dynamic_range.tflite	\
	custom_objects_float16.tflite		\
	custom_objects_float32.tflite		\
	custom_objects_int8_float_io.tflite	\
	custom_objects_int8.tflite

CAMERA_DIR	= /media/dgriffit/2821-03A1
CAMERA_TFLITE	= ${addprefix ${CAMERA_DIR}/,${MODELS}}

DST_DIR		= models
DST_TFLITE	= ${addprefix ${DST_DIR}/,${MODELS}}

NOTEBOOK	= assignment_4.ipynb

DOWNLOAD	= ~/Downloads

tocamera	: ${CAMERA_TFLITE}

fromdwnld	: ${DST_TFLITE}
	
${DST_DIR}/%	: ${DOWNLOAD}/%
	@mkdir -p ${dir $@}
	mv $^ $@

${CAMERA_DIR}/%	: ${DST_DIR}/%
	cp $^ $@


