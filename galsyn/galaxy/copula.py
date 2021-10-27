class Copula:

    def convert_magnitude_to_flux(self, plate_scale:float) -> None:
        """
        Converts the surface brightnesses of the galaxies in magnitudes / arcsecond²
        to flux/pixel. The flux values are inserted into the dataframe.

        Parameters
        ----------
        plate_scale : float
            The plate_scale scale of the image in arcseconds / pixel

        Examples
        --------
        from galsyn.galaxy.dataset import Gadotti

        galaxy = Gadotti()
        galaxy.sample(5)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        print(galaxy.data['flux_disk_r'])
        """
        components = self.get_components()

        for c in components:
            # Get all the magnitude columns
            magnitude_column = self.magnitude_column
            if isinstance(magnitude_column, dict):
                magnitude_column = magnitude_column[c]
            cols = [c for c in self.data.columns if c.startswith(magnitude_column)]

            # Convert to flux
            flux_column = self.flux_column
            if isinstance(self.flux_column, dict):
                flux_column = flux_column[c]
            for col in cols:
                filter_band = col.split('_')[-1]
                magnitude = self.data[col]
                flux = self.magnitude_to_flux(magnitude, filter_band) * plate_scale**2
                self.data[f'{flux_column}_{c}_{filter_band}'] = flux

    def get_components(self) -> set:
        """
        Returns the names of the galaxy components associated with the model. It
        assumes the columns are named with the following convention:

            {parameter}_{component}_{filter}

        and checks for the set of components associated with it.

        Returns
        -------
        components : set
            The various components in the model.

        Examples
        --------
        from galsyn.galaxy.dataset import Gadotti

        galaxy = Gadotti()
        galaxy.sample(5)
        components = galaxy.get_components()
        print(components)
        """
        if 'components' in self.__dict__:
            return self.components

        components = []
        for c in self.data.columns:
            components.append(c.split('_')[1])
        self.components = set(components)
        return self.components

    def sample(self, n:int) -> None:
        """
        Generates `n` samples from the copula and stores the samples in self.data

        Parameters
        ----------
        n : int
            The number of samples

        Examples
        --------
        from galsyn.galaxy.dataset import Gadotti

        galaxy = Gadotti()
        galaxy.sample(5)
        print(galaxy.data)
        """
        self.data = self.generator.sample(n)