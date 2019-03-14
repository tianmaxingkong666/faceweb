docker run \
	--name insightface \
	-v /opt/insightface:/opt/insightface \
	-v /opt/images:/opt/images \
	-p 5000:5000 \
	-e WORKERS=1 \
	-e THREADS=2 \
	-e PORT=5000 \
	-e PYTHONUNBUFFERED=0 \
	-d \
	insightface

