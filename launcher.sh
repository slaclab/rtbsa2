# /usr/bin/bash
if [[ $1 == "lcls" ]]; then
	HOSTNAME="physics@lcls-srv02"
elif [[ $1 == "facet" ]]; then
    HOSTNAME="fphysics@facet-srv02"
else
	echo " --> Bad arguments, must specify 'lcls' or 'facet'"
    exit 0
fi

ssh -Y $HOSTNAME "pydm --hide-nav-bar --hide-status-bar --hide-menu-bar /home/fphysics/zack/workspace/rtbsa2/rtbsaGUI.py"
