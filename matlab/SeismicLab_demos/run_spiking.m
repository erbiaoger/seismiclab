if exist('matlab_spiking_decon_demo.mat','file'), delete('matlab_spiking_decon_demo.mat'); end
cd('/Users/zhangzhiyu/MyProjects/SeismicLab/SeismicLab_demos');
try
  spiking_decon_demo;
  save('matlab_spiking_decon_demo.mat','s','o','Ps','Po','f');
catch ME
  disp(getReport(ME,'extended'));
end
