##
function test_statistics()
    ##
    name_data = "random"
    dir_save = "dir_save_test/statistics_test/"
    ##
    data = PDEHats.get_data("testing");
    trajectories = PDEHats.get_trajectories(data)
    #
    PDEHats.visualize_statistics_trajectories(trajectories; dir_save=dir_save)
    PDEHats.visualize_statistics_data(data; dir_save=dir_save)
    ##
    return true
end
##
