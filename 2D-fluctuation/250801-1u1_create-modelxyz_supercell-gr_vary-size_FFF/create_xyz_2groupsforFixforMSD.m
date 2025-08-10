clear; close all;

r0=[1/2,0,1/2;0,1/6,1/2;0,1/2,1/2;1/2,2/3,1/2];
n0=size(r0,1);
%nxyz=[36,21,1];   % 8.8 nm, L=500
%nxyz=[72,42,1];   % 17.65 nm, L=1000
%nxyz=[115,66,1];   % 28.24 nm, L=1600

%nxyz=[4,2,1];   % L=50 --> 8.8 A
%nxyz=[7,4,1];   % L=100 --> 17.65 A
%nxyz=[8,8,1];   % L=160 --> 28.24 A

%nxyz=[5,5,1];
%nxyz=[256,148,1];  % for 500 A x 500 A
%nxyz=[203,117,1];  % for 500 A x 500 A
%nxyz=[163,94,1];  % for 400 A x 400 A
%nxyz=[122,70,1];  % for 300 A x 300 A
%nxyz=[61,35,1];  % for 150 A x 150 A
%nxyzs={[256,148,1],[203,117,1],[163,94,1],[122,70,1],[61,35,1]};
nxyzs={[7,4,1],[35,20,1],[70,40,1],[105,60,1],[140,80,1],[175,100,1],[210,120,1]}; 
%nxyzs={[35,20,1]}; 
%nxyzs={[70,40,1]}; 
len_nxyzs=length(nxyzs);
for i=1:1:len_nxyzs
    nxyz=nxyzs{i};
    
    N=nxyz(1)*nxyz(2)*nxyz(3)*n0;
    a=[1.42*sqrt(3),1.42*3,20];
    r=zeros(N,3);

    label_0=zeros(N,1);    % for fix atoms
    label_1=zeros(N,1);     % for compute msd for atoms in specital zone
    center_nx=round(nxyz(1)./2 - 1);
    center_ny=round(nxyz(2)./2 - 1);

    for tolerance=[200]
        center_nx_lb=center_nx - tolerance;
        center_nx_rb=center_nx + tolerance;
        center_ny_lb=center_ny - tolerance;
        center_ny_rb=center_ny + tolerance;


        n=0;
        for ny=0:nxyz(2)-1
            for nx=0:nxyz(1)-1
                for nz=0:nxyz(3)-1
                    for m=1:n0
                        n=n+1;
                        r(n,:)=a.*([nx,ny,nz]+r0(m,:));
                        if (0<nx) && (nx<nxyz(1)-1) && (0<ny) && (ny<nxyz(2)-1)
                            label_0(n)=0;
                        else
                            label_0(n)=1;
                        end

                        %if nx==center_nx && ny==center_ny   %??????
                        if (nx>=center_nx_lb) && (nx<center_nx_rb) && (ny>=center_ny_lb) && (ny<center_ny_rb)
                            label_1(n)=1;
                        else
                            label_1(n)=0;
                        end
                    end
                end
            end
        end

        model_name=['model_size-',num2str(N),'.xyz'];
        fid=fopen(model_name,'w');
        fprintf(fid,'%d\n',N);
        fprintf(fid,'pbc=\"F F F\" Lattice=\"%g 0 0 0 %g 0 0 0 %g\" Properties=species:S:1:pos:R:3:group:I:2\n',a.*nxyz);

        for n=1:N
            fprintf(fid,'C %g %g %g %d %d\n',r(n,:),label_0(n),label_1(n));
        end
        fclose(fid);
        disp(['Size = ',num2str(N)]);
    end

end
