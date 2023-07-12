"""Tests for representing MIDI events and their properties."""
import pytest

from generative_music.domain.midi_data_processor.data_elements import (
    Item, ItemName)


class TestItem:
    """A test class for the Item class.

    The class tests if the Item instance is correctly created with valid and invalid names,
    and if the optional arguments are correctly assigned to the instance attributes.
    """

    def test_init_valid_name(self):
        """Check if the Item instance is correctly created with a valid name.

        This test checks if the name, start, end, velocity and pitch attributes
        are correctly assigned when the Item instance is created with a valid name.
        """
        item = Item(name=ItemName.NOTE, start=0)
        assert item.name == ItemName.NOTE
        assert item.start == 0
        assert item.end is None
        assert item.velocity is None
        assert item.pitch is None
        assert item.tempo is None

    def test_init_invalid_name(self):
        """Check if a ValueError is raised when creating an Item instance with an invalid name.

        This test checks if a ValueError is raised when the Item instance is created
        with an invalid name that is not a member of the ItemName Enum.
        """
        with pytest.raises(ValueError):
            Item(name="InvalidName", start=0)

    def test_init_optional_arguments(self):
        """Check if the Item instance is correctly created with optional arguments.

        This test checks if the end, velocity, and pitch attributes are correctly
        assigned when the Item instance is created with optional arguments.
        """
        item = Item(
            name=ItemName.NOTE, start=0, end=10, velocity=100, pitch=60, tempo=100
        )
        assert item.name == ItemName.NOTE
        assert item.start == 0
        assert item.end == 10
        assert item.velocity == 100
        assert item.pitch == 60
        assert item.tempo == 100
