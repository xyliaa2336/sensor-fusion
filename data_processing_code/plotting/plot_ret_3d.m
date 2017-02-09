%close all
s=get(0,'screensize');
row=3;
col=4;
% %Change the interested_interval to select the time segment to be plotted
% plot out all events
% interested_interval=[min(allTs_ret),max(allTs_ret)]; 
% plot out events within a certain interval
 interested_interval=[1,2.5];
dt=(interested_interval(2)-interested_interval(1))/(row*col);
fig_handle=zeros(1,row*col);

for row_cnt=1:row
    for col_cnt=1:col
        i=(row_cnt-1)*4+col_cnt;
        fig_handle(i)=figure('Position',[s(3)/col*(col_cnt-1), s(4)-s(4)/row*row_cnt, s(3)/col, s(4)/row]);
        
        indt=find((allTs_ret>((i-1)*dt+interested_interval(1))) & (allTs_ret<(i*dt+interested_interval(1))));
        plot3(allTs_ret(indt),x(indt),y(indt),'.');xlabel('t');ylabel('x');zlabel('y');

        title(['Nr.' num2str(i) 'Time slot: ' num2str(((i-1)*dt+interested_interval(1))) ' to ' num2str((i*dt+interested_interval(1)))])

        view(90,0)
    end
end

