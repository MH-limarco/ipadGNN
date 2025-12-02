python main.py --methood $1 --model $2 --missing_rate 0.0 --classic

python main.py --methood $1 --model $2 --missing_rate 0.3 --fill fp --classic
python main.py --methood $1 --model $2 --missing_rate 0.3 --fill apcfi --classic
python main.py --methood $1 --model $2 --missing_rate 0.3 --fill zero --classic

python main.py --methood $1 --model $2 --missing_rate 0.6 --fill fp --classic
python main.py --methood $1 --model $2 --missing_rate 0.6 --fill apcfi --classic
python main.py --methood $1 --model $2 --missing_rate 0.6 --fill zero --classic

python main.py --methood $1 --model $2 --missing_rate 0.9 --fill fp --classic
python main.py --methood $1 --model $2 --missing_rate 0.9 --fill apcfi --classic
python main.py --methood $1 --model $2 --missing_rate 0.9 --fill zero --classic

python main.py --methood $1 --model $2 --missing_rate 0.995 --fill fp --classic
python main.py --methood $1 --model $2 --missing_rate 0.995 --fill apcfi --classic
python main.py --methood $1 --model $2 --missing_rate 0.995 --fill zero --classic
