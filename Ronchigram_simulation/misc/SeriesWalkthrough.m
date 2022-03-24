for i = 1:100
    imagesc(ronch_final(:,:,i));colormap gray;axis equal off;
%     imagesc(chi0_final(:,:,i));colormap gray;axis equal off;
    pause(1)
end