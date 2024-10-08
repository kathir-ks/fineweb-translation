FROM python

WORKDIR /

COPY setup_inference_env.sh /

RUN chmod +x ./setup_inference_env.sh

RUN ./setup_inference_env.sh

COPY . .

CMD ["python3", "inference.py","--name", "HuggingFaceTB/cosmopedia", "--subset","khanacademy", "--batch_size", "256", "--bucket", "gs://indic-llama-data"]
