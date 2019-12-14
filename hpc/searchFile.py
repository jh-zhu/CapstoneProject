import os

dir_folder = '/scratch/yc3656/test/'
output_folder=dir_folder+'output3/'
sFile_pre=dir_folder+'main_script3/run'
_sigma=1
n_nodes=1
tpn=1

sfile=lambda i: sFile_pre+str(i)+'.s'
# sfile=lambda i: 'run'+str(i)+'.s'
n_file = 0
s_script = f'#!/bin/bash \n#SBATCH --nodes={n_nodes} \n#SBATCH --ntasks-per-node={tpn} \n#SBATCH --cpus-per-task=1 \n#SBATCH --time=20:00:00 \n#SBATCH --mem=2GB \n#SBATCH --job-name=runPython \n#SBATCH --error=expert_%A_%a.err \n\nmodule load python3/intel/3.7.3 \n\ncd '+dir_folder+'scripts3 \n'
if os.path.exists(sfile(n_file)):
    os.remove(sfile(n_file))

file_w=open(sfile(n_file),'w')
file_w.write(s_script)

n_line=1
file_indiv_line=1

sfiles_read=[dir_folder+'main_script3/run'+str(i)+'_'+str(j)+'.s' for i in range(21) for j in range(20) ]
#sfiles='run0_0.s' #'/scratch/yc3656/test/main_script3/run'
for sfile_read in sfiles_read:
    file=open(sfile_read,'r')
    pre=0
    for line in file.readlines():
        pre+=1
        if pre<13:
            continue
        line_element=line.split()
        if len(line_element)<7:
            continue
        model_name=line_element[7]
        model_parameters=line_element[8]
        n_random = line_element[9]
        n_experts=line_element[10]
        file_name = model_name + '_' + model_parameters + '.csv'
        output_directory = output_folder+'{}/{}/{}/'.format(_sigma,n_random,n_experts)
        output_file_path = output_directory + file_name


        if not os.path.exists(output_file_path):
            file_w.write(line)
            n_line+=1
            file_indiv_line+=1
            
        if n_line%31==0 and file_indiv_line>1:
            file_w.write('wait ')
            file_w.close()
            n_file+=1
            if os.path.exists(sfile(n_file)):
                os.remove(sfile(n_file))

            file_w=open(sfile(n_file),'w')
            file_indiv_line=1
            file_w.write(s_script)

print(n_file)
file_w.write('wait')
file_w.close()

