# -*- coding: utf-8 -*-

import numpy as np
from sys import exit, argv
import h5py as h5
import eagle_tools

from python_tools import *
plotconfig()

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib.colorbar import Colorbar

import statsmodels.nonparametric.smoothers_lowess as lowess

import safe_colours
safe_colours = safe_colours.initialise()
col_dict = safe_colours.distinct_named()
rainbow_inv = safe_colours.colourmap('rainbow',invert=True)
rainbow = safe_colours.colourmap('rainbow',invert=False)

C = eagle_tools.utilities.constants()


##########################################################################################################################################################
##########################################################################################################################################################
# Helper functions

def median_and_delta(xs,ys):

    running_median = lowess.lowess(ys, xs, frac=0.2, it=3, delta=0.0, is_sorted=True, missing='none', return_sorted=False)

    return running_median, ys-running_median

def trim(data,amount=10):
    return data[amount:-amount]

##########################################################################################################################################################
##########################################################################################################################################################

# Plotting functions

def coloured_scatter(catalogues,ax,rho_ax=None,
                        cmap=rainbow,
                        s=10,
                        lw=0.3,
                        show_xlabels=True,
                        show_ylabels=True,
                        label_size=22,
                        trim_median=10,
                        rho_window_sizes=[300,100],
                        rho_window_steps=[50,25],
                        rho_transition_points=[12.,]):
    '''
    Generate a standard plot combo of a coloured scatter and moving rho on a given ax and rho_ax.
    Computes the moving rank based on default settings for plots vs. M200 - can define other settings as kwargs.
    Can also set how many datapoints to trim from each end of the running median line.
    '''

    # Unpack the catalogues
    x, y, c = catalogues

    # Find the median y and c as a function of x, and the residuals
    ymed, ydel = median_and_delta(x['data'],y['data'])
    cmed, cdel = median_and_delta(x['data'],c['data'])

    # Make the scatter plot.
    scatter = ax.scatter(x['data'],y['data'],c=cdel,cmap=cmap,vmin=c['delta_limits'][0],vmax=c['delta_limits'][1],s=10,lw=0.3,edgecolor='gray',rasterized=True)
    ax.plot(trim(x['data'],amount=trim_median),trim(ymed),c='w',lw=4)
    ax.plot(trim(x['data'],amount=trim_median),trim(ymed),c='k',lw=2)

    # Set axis limits
    ax.set_ylim(y['limits'])
    ax.set_xlim(x['limits'])

    if show_ylabels:
        ax.set_ylabel(y['label'],fontsize=22)
    else:
        ax.set_yticklabels([])

    # Moving spearman rank axis
    if rho_ax is not None:
        # Hide the x labels from the main axis
        ax.set_xticklabels([])

        # Compute the moving spearman rank, return the window centres, ranks and p-values.
        rho_centres, rho, p = eagle_tools.plot.get_moving_spearman_rank(x['data'], y['data'], c['data'], window_sizes=rho_window_sizes, window_steps=rho_window_steps, transition_points=rho_transition_points)

        # Plot the running ranks
        rho_ax.axhline(0,c='gray',lw=1)
        rho_ax.plot(x['data'][rho_centres],rho,lw=2,c='k')

        rho_ax.set_xlim(x['limits'])
        rho_ax.set_ylim(-1.,1.)

        if not show_xlabels:
            rho_ax.set_xticklabels([])
        else:
            rho_ax.set_xlabel(x['label'],fontsize=label_size)

        if not show_ylabels:
            rho_ax.set_yticklabels([])
        else:
            rho_ax.set_ylabel(r'$\rho$',fontsize=label_size)

    else:
        # If no spearman rank axis, show the x label on the primary axis
        if show_xlabels:
            ax.set_xlabel(x['label'],fontsize=label_size)
        else:
            pass

    # Return the mappable scatter for plotting colourbars.
    return scatter


def add_colourbar(mappable,cbar_axis,
                    label='',
                    location='top',
                    label_size=22):

    assert location in ['top','bottom','left','right'],'Colourbar location must be top, bottom, left or right'

    if location == 'top':
        cbar = Colorbar(ax = cbar_axis, mappable = mappable, orientation = 'horizontal', ticklocation = 'top')
        cbar.set_label(label, labelpad=10,fontsize=label_size)

    elif location == 'bottom':
        cbar = Colorbar(ax = cbar_axis, mappable = mappable, orientation = 'horizontal', ticklocation = 'bottom')
        cbar.set_label(label, labelpad=10,fontsize=label_size)

    elif location == 'right':
        cbar = Colorbar(ax = cbar_axis, mappable = mappable, orientation = 'vertical', ticklocation = 'right')
        cbar.set_label(label, labelpad=10,fontsize=label_size)

    elif location == 'left':
        cbar = Colorbar(ax = cbar_axis, mappable = mappable, orientation = 'vertical', ticklocation = 'left')
        cbar.set_label(label, labelpad=10,fontsize=label_size)



##########################################################################################################################################################
##########################################################################################################################################################
# Catalogue loading and management

class catalogue(object):

    def __init__(self,sim = 'L0100N1504',
                        run = 'REFERENCE',
                        tag='028_z000p000'):

        self.sim = sim
        self.run = run
        self.tag = tag

        # Initialise an eagle_tools snapshot instance for loading easy things from FOF/SUBFIND
        self.Snapshot = eagle_tools.read.snapshot(sim=self.sim,run=self.run,tag=self.tag)


    def fof_match(self,groupnumbers,quantity):

        data = self.Snapshot.fof(quantity)

        return data[groupnumbers-1], []

    def subfind_match(self,groupnumbers,quantity):

        data = self.Snapshot.subfind(quantity)
        if quantity == 'ApertureMeasurements/Mass/030kpc':
            data = data[:,4]
        data = data[self.Snapshot.first_subhalo]

        return data[groupnumbers-1], []

    def catalogue_match(self,groupnumbers,quantity,filepath,gn_field='GroupNumber'):
        '''
        For loading in pre-computed catalogues by group number.
        Assumes that the group number is stored in the catalogue as 'GroupNumber', but this can be changed.
        Where the group is missing, returns nan.
        '''

        with h5.File(filepath, 'r') as f:
            file_groupnumbers = np.array(f[gn_field])                

            group_locations = np.zeros(len(groupnumbers),dtype=np.int64)
            missing = []

            for i, g in enumerate(groupnumbers):
                try:
                    group_locations[i] = np.where(file_groupnumbers==g)[0]
                except ValueError:
                    missing.append(i)

            loaded_data = np.array(f[quantity])[group_locations]

            # loaded_data[missing] = np.nan

        print 'Loaded ',quantity,', ',len(missing),' of ',len(groupnumbers),' objects are missing and should be cleaned up.'

        return loaded_data, missing



    def cleanup(self,xdict,ydict,cdict=None,remove=None):
        '''
        Remove any missing values from all datasets before plotting.
        Data for colouring is optional, in case of not colouring data points by anything.
        Can add extra indices as "remove" to remove extra data, such as where sSFR<10^-13
        '''

        print 'Cleaning up.'

        missing = xdict['missing']+ydict['missing']

        if cdict is not None:
            missing.extend(cdict['missing'])

        print 'Removing ',len(np.unique(missing)),' objects missing from one or more catalogues.'

        if remove is not None:
            missing.extend(remove)
            print 'Removing ',len(remove),' objects on request.'
            print 'Removing ',len(missing),' objects total.'

        xdict['data'] = np.delete(xdict['data'],missing)
        ydict['data'] = np.delete(ydict['data'],missing)

        print len(xdict['data']),' objects remain.'
        print '\n'
        
        del xdict['missing']
        del ydict['missing']
        
        if cdict is not None:
            del cdict['missing']
            cdict['data'] = np.delete(cdict['data'],missing)

        # Sort in ascending x
        xsort = np.argsort(xdict['data'])
        for key in ['groupnumbers','data']:
            xdict[key] = xdict[key][xsort]
            ydict[key] = ydict[key][xsort]
            if cdict is not None:
                cdict[key] = cdict[key][xsort]



        if cdict is not None:
            dict_list = [xdict,ydict,cdict]
        else:
            dict_list = [xdict,ydict]

        for d in dict_list:

            if d['limits'] is None:
                d['limits'] = [np.amin(d['data'][np.isfinite(d['data'])]),np.amax(d['data'][np.isfinite(d['data'])])]

            if d['delta_limits'] is None:

                print 'Pick your delta limits for ',d['name']

                mdict = self.load(d['groupnumbers'],'M200')
                
                plt.figure()
                plt.scatter(mdict['data'],d['data'],facecolor=None,edgecolor='k',s=5)
                plt.show()

                exit()



        if cdict is not None:     
            return xdict, ydict, cdict
        else:
            return xdict, ydict


    def get_data(self,gns,x_quantity,y_quantity,c_quantity=None):
        '''
        Wrapper function for quickly loading and preparing x, y and colour data in one go.
        '''

        x_cat = self.load(gns,x_quantity)
        y_cat = self.load(gns,y_quantity)

        if c_quantity is not None:
            c_cat = self.load(gns,c_quantity)
            return self.cleanup(x_cat,y_cat,c_cat)
        else:
            return self.cleanup(x_cat,y_cat)


    ####################################################################################################################################################################################
    # Define new quantities here!!!!


    def load(self,groupnumbers,quantity):


        if quantity == 'BlackHoleMass':

            loaded_data, missing = self.catalogue_match(groupnumbers,'MostMassiveBlackHoleMass',filepath='/hpcdata0/arijdav1/halo_data/'+self.sim+'_'+self.run+'_'+self.tag+'_mostmassiveBH.hdf5')
            loaded_data = np.log10(loaded_data)

            label = r'$\log_{10}(M_{\mathrm{BH}})\,[\mathrm{M}_{\odot}]$'
            delta_label = r'$\Delta \log_{10}(M_{\mathrm{BH}})$'
            limits = None
            delta_limits = [-0.75,0.75]

        
        elif quantity == 'Mstar':

            loaded_data, missing = self.subfind_match(groupnumbers,'ApertureMeasurements/Mass/030kpc')
            loaded_data = np.log10(loaded_data*1e10)

            label = r'$\log_{10}(M_{*})\,[\mathrm{M}_{\odot}]$'
            delta_label = r'$\Delta \log_{10}(M_{*})$'
            limits = [9.,12.]
            delta_limits = [-0.3,0.3]


        elif quantity == 'M200':

            loaded_data, missing = self.fof_match(groupnumbers,'Group_M_Crit200')
            loaded_data = np.log10(loaded_data*1e10)

            label = r'$\log_{10}(M_{200})\,[\mathrm{M}_{\odot}]$'
            delta_label = r'$\Delta \log_{10}(M_{200})$'
            limits = [11.3,14.8]
            delta_limits = [-1.,1.]


        elif quantity == 'f_CGM':

            loaded_data, missing = self.catalogue_match(groupnumbers,'f_CGM_nosf',filepath='/hpcdata0/arijdav1/EAGLE_catalogues/'+self.sim+'/'+self.run+'/'+self.tag+'/halo_baryon_census.hdf5')

            label = r'$f_{\mathrm{CGM}}/(\Omega_{\mathrm{b}}/\Omega_0)$'
            delta_label = r'$\Delta f_{\mathrm{CGM}}/(\Omega_{\mathrm{b}}/\Omega_0)$'
            delta_limits = [-0.24,0.24]
            limits = [-0.05,1.05]


        elif quantity == 'f_ISM':

            loaded_data, missing = self.catalogue_match(groupnumbers,'f_ISM_sf_or_highdensity',filepath='/hpcdata0/arijdav1/EAGLE_catalogues/'+self.sim+'/'+self.run+'/'+self.tag+'/halo_baryon_census.hdf5')

            label = r'$f_{\mathrm{ISM}}/(\Omega_{\mathrm{b}}/\Omega_0)$'
            delta_label = r'$\Delta f_{\mathrm{ISM}}/(\Omega_{\mathrm{b}}/\Omega_0)$'
            delta_limits = [-0.015,0.015]
            limits = [-0.005,0.075]



        elif quantity == 'M_ISM':

            f_ISM, missing = self.catalogue_match(groupnumbers,'f_ISM_sf_or_highdensity',filepath='/hpcdata0/arijdav1/EAGLE_catalogues/'+self.sim+'/'+self.run+'/'+self.tag+'/halo_baryon_census.hdf5')

            M200, blah = self.fof_match(groupnumbers,'Group_M_Crit200')

            loaded_data = np.log10(f_ISM * M200 * 1e10 * self.Snapshot.f_b_cosmic)

            label = r'$\log_{10}(M_{\mathrm{ISM}})\,[\mathrm{M}_\odot]$'
            delta_label = r'$\Delta \log_{10}(M_{\mathrm{ISM}})$'
            delta_limits = [-1.,1.]
            limits = None






        elif quantity == 'E_bind_baryon_DMO':

            loaded_data, missing = self.catalogue_match(groupnumbers,'E_bind_baryon_DMO',filepath='/hpcdata0/arijdav1/EAGLE_catalogues/'+self.sim+'/'+self.run+'/'+self.tag+'/DMO_binding_energies.hdf5')
            
            # Add failed or unbound DMO matches to "missing"
            missing.extend(np.where(loaded_data>=0.)[0].tolist())

            loaded_data = np.log10(-1.*loaded_data)

            label = r'$\log(E_{\rm bind}^{\rm b})\,[{\rm erg}]$'
            delta_label = r'$\Delta \log(E_{\rm bind}^{\rm b})$'
            delta_limits = [-0.2,0.2]
            limits = None


        elif quantity == 'Vmax_over_V200_DMO':

            Ebind, missing = self.catalogue_match(groupnumbers,'E_bind_baryon_DMO',filepath='/hpcdata0/arijdav1/EAGLE_catalogues/'+self.sim+'/'+self.run+'/'+self.tag+'/DMO_binding_energies.hdf5')

            loaded_data, missing = self.catalogue_match(groupnumbers,'Vmax_over_V200_DMO',filepath='/hpcdata0/arijdav1/EAGLE_catalogues/'+self.sim+'/'+self.run+'/'+self.tag+'/DMO_binding_energies.hdf5')
            
            # Add failed or unbound DMO matches to "missing"
            missing.extend(np.where(Ebind>=0.)[0].tolist())

            label = r'$V_{\rm DMO}^{\rm max}/V_{\rm DMO}^{200}$'
            delta_label = r'$\Delta (V_{\rm DMO}^{\rm max}/V_{\rm DMO}^{200})$'
            delta_limits = [-0.16,0.16]
            limits = None


        elif quantity == 'concentration':

            conc, missing = self.catalogue_match(groupnumbers,'Concentration',filepath='/hpcdata0/arijdav1/EAGLE_catalogues/'+self.sim+'/'+self.run+'/'+self.tag+'/concentrations.hdf5')

            # Add failed or unbound DMO matches to "missing"
            missing.extend(np.where((conc==0.)|(np.isfinite(conc)==False))[0].tolist())

            loaded_data = 10.**conc

            label = r'$c$'
            delta_label = r'$\Delta c$'
            delta_limits = [-5,5.]
            limits = None








        elif quantity == 'connectivity_5sigma':

            loaded_data, missing = self.catalogue_match(groupnumbers,'S5/Connectivity',gn_field='S5/GroupNumber',filepath='/hpcdata0/arijdav1/cosmic_web/matched_catalogues/closest_nodes.hdf5')
            distance_to_node, missing = self.catalogue_match(groupnumbers,'S5/DistanceToNode',gn_field='S5/GroupNumber',filepath='/hpcdata0/arijdav1/cosmic_web/matched_catalogues/closest_nodes.hdf5')
            r200, missing = self.catalogue_match(groupnumbers,'S5/r200',gn_field='S5/GroupNumber',filepath='/hpcdata0/arijdav1/cosmic_web/matched_catalogues/closest_nodes.hdf5')

            # Add systems where node is outside r200 to "missing"
            missing.extend(np.where(distance_to_node>r200)[0].tolist())

            label = r'${\rm Connectivity}$'
            delta_label = r'$\Delta {\rm Connectivity}$'
            delta_limits = [-5,5]
            limits = None


        elif quantity == 'multiplicity_5sigma':

            loaded_data, missing = self.catalogue_match(groupnumbers,'S5/Multiplicity',gn_field='S5/GroupNumber',filepath='/hpcdata0/arijdav1/cosmic_web/matched_catalogues/closest_nodes.hdf5')
            distance_to_node, missing = self.catalogue_match(groupnumbers,'S5/DistanceToNode',gn_field='S5/GroupNumber',filepath='/hpcdata0/arijdav1/cosmic_web/matched_catalogues/closest_nodes.hdf5')
            r200, missing = self.catalogue_match(groupnumbers,'S5/r200',gn_field='S5/GroupNumber',filepath='/hpcdata0/arijdav1/cosmic_web/matched_catalogues/closest_nodes.hdf5')

            # Add systems where node is outside r200 to "missing"
            missing.extend(np.where(distance_to_node>r200)[0].tolist())

            label = r'${\rm Multiplicity}$'
            delta_label = r'$\Delta {\rm Multiplicity}$'
            delta_limits = [-2,2]
            limits = None


        elif quantity == 'distance_to_node':

            distance_to_node, missing = self.catalogue_match(groupnumbers,'S5/DistanceToNode',gn_field='S5/GroupNumber',filepath='/hpcdata0/arijdav1/cosmic_web/matched_catalogues/closest_nodes.hdf5')
            r200, missing = self.catalogue_match(groupnumbers,'S5/r200',gn_field='S5/GroupNumber',filepath='/hpcdata0/arijdav1/cosmic_web/matched_catalogues/closest_nodes.hdf5')

            loaded_data = distance_to_node/r200

            label = r'$d_{\rm node}/r_{200}$'
            delta_label = r'$\Delta d_{\rm node}/r_{200}$'
            delta_limits = [-0.1,0.1]
            limits = [-0.1,1.0]











        
        
        return {'name':quantity,'groupnumbers':groupnumbers,'data':loaded_data,'missing':missing,'label':label,'delta_label':delta_label,'limits':limits,'delta_limits':delta_limits}



####################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################









####################################################################################
# Setup

plot_name = argv[1]

save_directory = '/home/arijdav1/figures/coloured_scatter_plots/'

cat = catalogue()

# Define the sample
gns = cat.Snapshot.groupnumbers[np.log10(cat.Snapshot.M200 * 1e10)>11.5]

####################################################################################




if plot_name == 'Vmax_Ebind':

    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='Vmax_over_V200_DMO',c_quantity='E_bind_baryon_DMO')
    x, y, c = xyc

    plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(3,1, height_ratios=[0.05,1,0.2], width_ratios=[1,])
    gs.update(wspace=0., hspace=0.)

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])

    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)

    add_colourbar(scatter,cax,label=c['delta_label'])




elif plot_name == 'f_ISM_Ebind':

    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='f_ISM',c_quantity='E_bind_baryon_DMO')
    x, y, c = xyc

    plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(3,1, height_ratios=[0.05,1,0.2], width_ratios=[1,])
    gs.update(wspace=0., hspace=0.)

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])

    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)

    add_colourbar(scatter,cax,label=c['delta_label'])


elif plot_name == 'f_ISM_MBH':

    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='f_ISM',c_quantity='BlackHoleMass')
    x, y, c = xyc

    plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(3,1, height_ratios=[0.05,1,0.2], width_ratios=[1,])
    gs.update(wspace=0., hspace=0.)

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])

    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)

    add_colourbar(scatter,cax,label=c['delta_label'])


elif plot_name == 'f_ISM_Vmax':

    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='f_ISM',c_quantity='Vmax_over_V200_DMO')
    x, y, c = xyc

    plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(3,1, height_ratios=[0.05,1,0.2], width_ratios=[1,])
    gs.update(wspace=0., hspace=0.)

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])

    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)

    add_colourbar(scatter,cax,label=c['delta_label'])


elif plot_name == 'f_CGM_Vmax':

    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='f_CGM',c_quantity='Vmax_over_V200_DMO')
    x, y, c = xyc

    plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(3,1, height_ratios=[0.05,1,0.2], width_ratios=[1,])
    gs.update(wspace=0., hspace=0.)

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])

    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)

    add_colourbar(scatter,cax,label=c['delta_label'])


elif plot_name == 'multi_panel_test':

    plt.figure(figsize=(16,10))
    gs = gridspec.GridSpec(5,3, height_ratios=[0.05,1,0.2,1.,0.2], width_ratios=[1,1,1])
    gs.update(wspace=0., hspace=0.)

    # First row - f_ISM  - colourbars on top so need to add them here

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])
    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='f_CGM',c_quantity='BlackHoleMass')
    x, y, c = xyc
    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax,show_xlabels=False)
    add_colourbar(scatter,cax,label=c['delta_label'])

    ax = plt.subplot(gs[1,1])
    rho_ax = plt.subplot(gs[2,1])
    cax = plt.subplot(gs[0,1])
    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='f_CGM',c_quantity='E_bind_baryon_DMO')
    x, y, c = xyc
    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax,show_ylabels=False,show_xlabels=False)
    add_colourbar(scatter,cax,label=c['delta_label'])

    ax = plt.subplot(gs[1,2])
    rho_ax = plt.subplot(gs[2,2])
    cax = plt.subplot(gs[0,2])
    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='f_CGM',c_quantity='Vmax_over_V200_DMO')
    x, y, c = xyc
    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax,show_ylabels=False,show_xlabels=False)
    add_colourbar(scatter,cax,label=c['delta_label'])

    # Second row - f_CGM - no colourbars needed

    ax = plt.subplot(gs[3,0])
    rho_ax = plt.subplot(gs[4,0])
    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='f_CGM',c_quantity='BlackHoleMass')
    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)

    ax = plt.subplot(gs[3,1])
    rho_ax = plt.subplot(gs[4,1])
    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='f_CGM',c_quantity='E_bind_baryon_DMO')
    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax,show_ylabels=False)

    ax = plt.subplot(gs[3,2])
    rho_ax = plt.subplot(gs[4,2])
    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='f_CGM',c_quantity='Vmax_over_V200_DMO')
    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax,show_ylabels=False)




elif plot_name == 'connectivity_test':

    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='f_CGM',c_quantity='connectivity_5sigma')
    x, y, c = xyc

    plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(3,1, height_ratios=[0.05,1,0.2], width_ratios=[1,])
    gs.update(wspace=0., hspace=0.)

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])

    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)

    add_colourbar(scatter,cax,label=c['delta_label'])


elif plot_name == 'multiplicity_test':

    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='f_CGM',c_quantity='multiplicity_5sigma')
    x, y, c = xyc

    plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(3,1, height_ratios=[0.05,1,0.2], width_ratios=[1,])
    gs.update(wspace=0., hspace=0.)

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])

    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)

    add_colourbar(scatter,cax,label=c['delta_label'])



elif plot_name == 'multiplicity_BH':

    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='BlackHoleMass',c_quantity='multiplicity_5sigma')
    x, y, c = xyc

    plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(3,1, height_ratios=[0.05,1,0.2], width_ratios=[1,])
    gs.update(wspace=0., hspace=0.)

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])

    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)

    add_colourbar(scatter,cax,label=c['delta_label'])


elif plot_name == 'multiplicity_Ebind':

    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='E_bind_baryon_DMO',c_quantity='multiplicity_5sigma')
    x, y, c = xyc

    plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(3,1, height_ratios=[0.05,1,0.2], width_ratios=[1,])
    gs.update(wspace=0., hspace=0.)

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])

    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)

    add_colourbar(scatter,cax,label=c['delta_label'])



elif plot_name == 'distance_to_node':

    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='distance_to_node',c_quantity='multiplicity_5sigma')
    x, y, c = xyc

    plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(3,1, height_ratios=[0.05,1,0.2], width_ratios=[1,])
    gs.update(wspace=0., hspace=0.)

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])

    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)

    add_colourbar(scatter,cax,label=c['delta_label'])




elif plot_name == 'M_ISM_M_BH':

    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='M_ISM',c_quantity='BlackHoleMass')
    x, y, c = xyc

    plt.figure(figsize=(7,8))
    gs = gridspec.GridSpec(3,1, height_ratios=[0.05,1,0.2], width_ratios=[1,])
    gs.update(wspace=0., hspace=0.)

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])

    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)

    add_colourbar(scatter,cax,label=c['delta_label'])





elif plot_name == 'M_ISM_assembly_proxies':

    plt.figure(figsize=(16,10))
    gs = gridspec.GridSpec(5,3, height_ratios=[0.05,1,0.2,1.,0.2], width_ratios=[1,1,1])
    gs.update(wspace=0., hspace=0.)

    # First row - f_ISM  - colourbars on top so need to add them here

    ax = plt.subplot(gs[1,0])
    rho_ax = plt.subplot(gs[2,0])
    cax = plt.subplot(gs[0,0])
    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='M_ISM',c_quantity='E_bind_baryon_DMO')
    x, y, c = xyc
    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax)
    add_colourbar(scatter,cax,label=c['delta_label'])

    ax = plt.subplot(gs[1,1])
    rho_ax = plt.subplot(gs[2,1])
    cax = plt.subplot(gs[0,1])
    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='M_ISM',c_quantity='Vmax_over_V200_DMO')
    x, y, c = xyc
    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax,show_ylabels=False)
    add_colourbar(scatter,cax,label=c['delta_label'])

    ax = plt.subplot(gs[1,2])
    rho_ax = plt.subplot(gs[2,2])
    cax = plt.subplot(gs[0,2])
    xyc = cat.get_data(gns,x_quantity='M200',y_quantity='M_ISM',c_quantity='concentration')
    x, y, c = xyc
    scatter = coloured_scatter(xyc,ax,rho_ax=rho_ax,show_ylabels=False)
    add_colourbar(scatter,cax,label=c['delta_label'])










else:
    print 'Which plot?'
    exit()


plt.savefig(save_directory+plot_name+'.pdf',bbox_inches='tight',dpi=200)
