# following the tutorial in https://alien.slackbook.org/blog/remote-access-to-your-vnc-server-via-modern-browsers/

cd ~ || exit

curl -O https://github.com/novnc/noVNC/archive/refs/tags/v1.4.0.tar.gz
tar -C /usr/local -xvf v1.4.0.tar.gz
ln -s /usr/local/noVNC-1.4.0 /usr/local/novnc
cd /usr/local/novnc/utils/ || exit

git clone https://github.com/novnc/websockify websockify

# Start the client as below
# cd /usr/local/novnc/
#  ./utils/novnc_proxy --vnc localhost:5901
