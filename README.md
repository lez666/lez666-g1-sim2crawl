`python scripts/rsl_rl/train.py --task g1-crawl-proc --headless`

`python scripts/rsl_rl/train.py --headless`

`python scripts/rsl_rl/play.py --task g1-crawl-proc --checkpoint "/home/logan/Projects/g1_crawl/logs/rsl_rl/g1_crawl_procbase/2025-09-22_09-45-34/model_1499.pt"`


`python scripts/rsl_rl/train.py --task g1-crawl-transition`

`python scripts/rsl_rl/play.py --task g1-crawl-transition --headless --video --video_length 200 --enable_cameras`

`python scripts/rsl_rl/play.py --task g1-crawl-transition --checkpoint /home/logan/Projects/g1_crawl/logs/rsl_rl/g1_crawl_proc_simple/2025-10-01_07-40-45/model_2499.pt --headless --video --video_length 200 --enable_cameras`

python scripts/rsl_rl/play.py --task g1-crawl-transition --checkpoint /home/logan/Projects/g1_crawl/logs/rsl_rl/g1_crawl_proc_simple/2025-10-01_06-59-16/model_2499.pt


/home/logan/Projects/g1_crawl/logs/rsl_rl/g1_crawl_trans_simple/2025-10-01_11-19-59/model_2499.pt

python scripts/rsl_rl/play.py --task g1-crawl-transition --checkpoint /home/logan/Projects/g1_crawl/logs/rsl_rl/g1_crawl_trans_simple/2025-10-01_11-19-59/model_2499.pt --headless --video --video_length 200 --enable_cameras`


python scripts/rsl_rl/play.py --task g1-crawl-transition --checkpoint /home/logan/Projects/g1_crawl/logs/rsl_rl/g1_crawl_trans_simple/2025-10-01_10-49-02/model_2499.pt --headless --video --video_length 200 --enable_cameras

python scripts/rsl_rl/train.py \
  --task g1-crawl-start \
  --gui \
  agent.resume=false \
  agent.load_run=/home/logan/Projects/g1_crawl/logs/rsl_rl/g1_crawl_start/2025-10-17_07-58-54    \
  agent.load_checkpoint=model_1999.pt 

  python scripts/rsl_rl/train.py \
  --task g1-crawl-start 
  --resume_checkpoint /home/logan/Projects/g1_crawl/logs/rsl_rl/g1_crawl_start/2025-10-17_07-58-54/model_1999.pt 
