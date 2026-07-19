import redis
import redis.exceptions
from app.streaming.config import REDIS_HOST, REDIS_PORT
from core.logging import logger

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def ensure_group(stream_name, group_name):
    try:
        r.xgroup_create(name=stream_name, groupname=group_name, id="0", mkstream=True)
        logger.info(
            f"Created consumer group '{group_name}' for stream '{stream_name}'"
        )
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info(f"Consumer group '{group_name}' already exists for stream '{stream_name}'")
        else:
            logger.exception(f"Failed to create consumer group '{group_name}'")
            raise  # group already exists, safe to ignore


def push_job(stream_name, job_id):
    message_id = r.xadd(stream_name, {"job_id": job_id})
    logger.info(f"Pushed job={job_id} to stream='{stream_name}', message_id={message_id}")

def run_worker_loop(stream_name, group_name, consumer_name, next_stream, step_fn):
    ensure_group(stream_name, group_name)
    logger.info(
        f"Worker started | stream={stream_name} "
        f"group={group_name} consumer={consumer_name}"
    )
    while True:
        result = r.xreadgroup(
            groupname=group_name,
            consumername=consumer_name,
            streams={stream_name: ">"},
            count=1,
            block=5000,  # wait up to 5s for new messages
        )

        if not result or not isinstance(result, list):
            logger.debug(
                f"No messages available | stream={stream_name}"
            )
            continue

        for _, entries in result:
            for message_id, fields in entries:
                job_id = fields["job_id"]
                logger.info(
                    f"Received job={job_id} "
                    f"message_id={message_id} "
                    f"consumer={consumer_name}"
                )
                try:
                    logger.info(
                        f"Processing job={job_id}"
                        f"message_id={message_id} "
                        f"consumer={consumer_name}"
                    )
                    step_fn(job_id)  # do the actual work + write to Mongo
                    logger.info(
                        f"Completed job={job_id}"
                        f"message_id={message_id}"
                        f"consumer={consumer_name}"
                    )
                    if next_stream:
                        push_job(next_stream, job_id)
                    r.xack(stream_name, group_name, message_id)
                    logger.info(
                        f"Acknowledged job={job_id} "
                        f"message_id={message_id}"
                    )
                except Exception as e:
                    print(f"[{stream_name}] job {job_id} failed: {e}")
                    # deliberately NOT acking — message stays pending,
                    # gets picked up later by an XAUTOCLAIM sweep