import nwfunc
import subprocess as sp

while True:
    udo = input ('Press enter to continue')
    if udo == '':
        operation = 'ge'
        if operation == 'ge':
            xyzorg = 'pentacene.xyz'
            nwfolder = 'pentacene_'+input ('Type in folder name to store the files: ')
            [xstart,xstop,ystart,ystop,zstart,zstop] = list(map(float,input('Type in start and stop values, separated by a space (xstart xstop ystart ystop zstart zstop): ').split(' ')))
            space = 0.2
            [theta,phi,gamma] = [0,0,0]
            #[theta,phi,gamma] = list(map(float, input('rotation for the second molecule? (about the x,y,and z axis): ').split(' ')))
            ftype = 'nw'
            count = nwfunc.generate_file (xyzorg,nwfolder,xstart,xstop,ystart,ystop,zstart,zstop,space,FileType=ftype,theta=theta,phi=phi,gamma=gamma)
            print ('{} files generated. Do you want to run the calculation?'.format (count))
            boolcal = input ('"yes" or "no": ')
            if boolcal == 'yes':
                sp.run ('cd /pscratch/anthony_uksr/hmle231/NWchem/calculation_data/{}; sbatch $HOME/scripts/nwcbatch.sh'.format (nwfolder), shell=True)
            else:
                print ('Task completed')
        elif operation == 'pr':
            nwfolder = input ('Enter the folder containing the log files: ')
            result = nwfunc.log_process (nwfolder)
            print ('Task completed. Go to /pscratch/anthony_uksr/hmle231/NWchem/processed_csv/ to see the result.')
            
            
