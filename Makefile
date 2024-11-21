setup:
	rm -rf venv
	python3 -m venv venv
	pip install --upgrade pip
	venv/bin/pip install -r requirements.txt
	$(MAKE) dbu

dbu:
	pg_ctl -D /opt/homebrew/var/postgresql@14 start
	
dbd:
	pg_ctl -D /opt/homebrew/var/postgresql@14 stop

dbstatus:
	pgrep postgres

riqh:
	python3 -c "import os; os.chdir('.'); exec(open('Code/Rolling Intrinsic/Rolling Intrinsic QH.py').read())"

rih:
	python3 -c "import os; os.chdir('.'); exec(open('Code/Rolling Intrinsic/Rolling Intrinsic H.py').read())"

