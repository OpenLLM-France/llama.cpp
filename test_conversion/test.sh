cd `dirname $0`


WHAT="/mnt/d/home/jlouradour/luciole/releases/SFT/Luciole-1B-Instruct-1.0"
python test_main.py $WHAT $WHAT-gguf
