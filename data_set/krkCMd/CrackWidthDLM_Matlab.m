% The script is assessing multiple crack widths based on brightness profiles, with the use of a deep learning metasensor proposed by the authors [1,2]. 
% This is an illustrative script and benchmark data, for possible comparisons with user's own models. 
% [1] Jakubowski, J. & Tomczak, K. Deposition of data for developing deep learning models to assess crack width and self-healing progress in concrete (krkCMd). Zenodo doi: 10.5281/zenodo.11408398 (2024)
% [2] Jakubowski, J. & Tomczak, K. Deep learning metasensor for crack-width assessment and self-healing evaluation in concrete. Constr. Build. Mater. 422, 135768 (2024).
% This work is licensed under CC BY 4.0 

clear
Bet=readtable('krk_data.csv');
net=load('krknet.mat');
krknet=net.krknet;
XBet = Bet{:,7:507}; 
XBet = reshape(XBet',501,1,1,height(Bet));
pi2mi=1000*25.4/6400;
DLMwidth=pi2mi*predict(krknet,XBet);
DLMwidth(DLMwidth<0)=0;
DLMwidth = single(DLMwidth);
save('DLMwidth','DLMwidth');