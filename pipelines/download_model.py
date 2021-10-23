import boto3, os
      
s3 = boto3.resource("s3")
bucket = s3.Bucket("ai-tokenizer")
for obj in bucket.objects.all():
    if not os.path.exists(os.path.dirname(obj.key)):
        os.makedirs(os.path.dirname(obj.key))
    bucket.download_file(obj.key, obj.key)