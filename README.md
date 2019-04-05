# CS3244 Project - Birdwatching


## Quick Start
1. Install gcloud CLI (https://cloud.google.com/sdk/)
2. Go to Google Cloud Console > Select project `cs3244-ml` > Compute Engine > VM Instances > Start `cs3244-ml-instance`	
3. On your local machine, use ssh port forwarding to connect to the remote instance.

    > gcloud compute ssh --zone=asia-southeast1-b jupyter@cs3244-ml-instance -- -L 9000:localhost:8080

4. Open `localhost:9000` in the browser.
5. IMPORTANT: stop the instance when you have done the work. The instance is charged US$0.864 hourly.

See fast.ai's setup guide https://course.fast.ai/start_gcp.html) if you encounter any issues.


## Instance Type

n1-highmem-8 (8 vCPUs, 52 GB memory) with 1 x NVIDIA Tesla P4

## Useful Commands
```
# monitor GPU usage
> watch -n 0.5 nvidia-smi

```

## Trouble Shoot
### RuntimeError: CUDA out of memory

Usually happens when a training task is forcibly stopped.
Solution: `Menu > Kernel > Restart kernel`.